"""Foundation-model pseudo-labelling: DINOv2 patch features + clustering.

This is the lightweight, file-in/file-out labeller behind ``client.labeling.foundation``. It embeds an RGB
rendering of the scene with a DINOv2 backbone, clusters the patch embeddings and paints each cluster id
back onto the pixels its patch covers. The result is an *unsupervised* segmentation: cluster ids carry no
class names (use :mod:`pygeovision.labeling.quality` / manual inspection to name them).

For multispectral models (Prithvi, RemoteCLIP text prompts) use
:class:`pygeovision.ai.labeling.foundation_labeler.FoundationModelLabeler` instead.

Registration
------------
The scene is cut into ``tile_size`` windows that are fed to the backbone *without* the processor's
resize / centre-crop, so patch ``(i, j)`` of a window covers exactly pixels ``[i*p, (i+1)*p) x [j*p, (j+1)*p)``
of that window. (The previous version let the processor resize the whole scene to 256 px and centre-crop
224 px, so the label grid covered only the middle of the image and was then stretched over all of it.)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NODATA_LABEL = 255
MAX_CELLS = 1_500_000        # ~4.5 GB of float32 for a 768-d backbone; refuse rather than thrash

HF_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
}


class FoundationModelLabeler:
    """Pseudo-labels from DINOv2 patch features + clustering.

    Parameters
    ----------
    model : str
        ``dinov2-small|base|large|giant`` or any Hugging Face DINOv2-style repo id.
    device : str, optional
        ``"cuda"``, ``"cpu"`` ... Defaults to CUDA when available.
    rgb_bands : (int, int, int), optional
        1-based raster bands rendered as R, G, B. Defaults to ``(1, 2, 3)`` (or the single band repeated
        for one-band rasters). Sentinel-2 files in B2,B3,B4 order need ``(3, 2, 1)``.

    Example
    -------
    >>> lab = FoundationModelLabeler(model="dinov2-large", rgb_bands=(3, 2, 1))
    >>> lab.pseudo_label("scene.tif", "pseudo.tif", n_classes=8)["success"]
    True
    """

    def __init__(self, model: str = "dinov2-base", device: str | None = None,
                 rgb_bands: tuple[int, int, int] | None = None) -> None:
        self.model_name = model
        self.device = device
        if rgb_bands is not None and len(rgb_bands) != 3:
            raise ValueError("rgb_bands must name exactly three bands (R, G, B)")
        self.rgb_bands = tuple(rgb_bands) if rgb_bands else None
        self._loaded: tuple[Any, tuple[float, ...], tuple[float, ...], int] | None = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> tuple[Any, tuple[float, ...], tuple[float, ...], int]:
        """Return ``(model, mean, std, patch_size)``; raises on failure (nothing is substituted)."""
        if self._loaded is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            hf_id = HF_IDS.get(self.model_name, self.model_name)
            dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            processor = AutoImageProcessor.from_pretrained(hf_id)
            model = AutoModel.from_pretrained(hf_id).to(dev).eval()
            patch = getattr(model.config, "patch_size", None)
            if not isinstance(patch, int):
                raise RuntimeError(f"{hf_id} does not expose an integer config.patch_size; "
                                   "it is not a ViT-style backbone this labeller can tile")
            self._loaded = (model, tuple(processor.image_mean), tuple(processor.image_std), patch)
        return self._loaded

    # ------------------------------------------------------------------ main entry
    def pseudo_label(
        self,
        image_path: str | Path,
        output_path: str | Path,
        n_classes: int = 8,
        patch_size: int | None = None,
        clustering: str = "kmeans",  # kmeans | hdbscan
        clip_annotate: bool = False,
        tile_size: int = 448,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Cluster DINOv2 patch features and write the cluster map as a GeoTIFF.

        Parameters
        ----------
        n_classes : int
            Number of k-means clusters (ignored by HDBSCAN, which finds its own).
        patch_size : int, optional
            Must match the backbone's patch size if given; read from the model otherwise.
        clustering : {"kmeans", "hdbscan"}
        clip_annotate : bool
            Naming clusters with CLIP is not implemented; ``True`` raises ``NotImplementedError``
            (it used to be accepted and silently ignored).
        tile_size : int
            Window fed to the backbone, rounded down to a multiple of the patch size.

        Returns
        -------
        dict
            ``success``, ``output_path``, ``n_classes``, ``n_patches``, ``cell_px``, ``valid_fraction``,
            ``model``; or ``success=False`` and ``error``.
        """
        if clip_annotate:
            raise NotImplementedError(
                "clip_annotate is not implemented: clusters are unnamed. Name them yourself or use "
                "pygeovision.ai.labeling.foundation_labeler.FoundationModelLabeler(model='remote_clip', "
                "text_prompts=[...]) for prompt-based labels."
            )
        if clustering not in ("kmeans", "hdbscan"):
            return {"success": False, "error": f"Unknown clustering: {clustering}"}
        if n_classes < 2 and clustering == "kmeans":
            raise ValueError("n_classes must be >= 2")

        output_path = Path(output_path)

        try:
            import numpy as np
            import rasterio
            import torch
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        try:
            model, mean, std, patch = self._load()
        except Exception as exc:  # noqa: BLE001
            return {"success": False,
                    "error": f"could not load '{self.model_name}': {type(exc).__name__}: {exc}. "
                             "Nothing was substituted."}
        if patch_size is not None and patch_size != patch:
            raise ValueError(f"patch_size={patch_size} but {self.model_name} uses {patch}")

        tile = max(patch, (int(tile_size) // patch) * patch)
        per = tile // patch

        # ---- read + render RGB (nodata-aware, one global stretch so tiles stay comparable)
        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            bands = list(self.rgb_bands or ((1, 2, 3) if src.count >= 3 else (1, 1, 1)))
            if max(bands) > src.count or min(bands) < 1:
                return {"success": False,
                        "error": f"rgb_bands {tuple(bands)} not available in a {src.count}-band raster"}
            data = src.read(bands).astype(np.float32)
            valid = src.read_masks(bands[0]) > 0
            for b in bands[1:]:
                valid &= src.read_masks(b) > 0
            valid &= np.isfinite(data).all(0)
        if not valid.any():
            return {"success": False, "error": "raster has no valid pixels"}

        rgb = np.zeros_like(data)
        for k in range(3):
            lo, hi = np.percentile(data[k][valid], (2, 98))
            rgb[k] = np.clip((data[k] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        rgb[:, ~valid] = 0.0
        mean_a = np.asarray(mean, np.float32)[:, None, None]
        std_a = np.asarray(std, np.float32)[:, None, None]

        rows, cols = -(-height // patch), -(-width // patch)
        if rows * cols > MAX_CELLS:
            return {"success": False,
                    "error": f"{rows * cols} patches exceeds the {MAX_CELLS} limit; "
                             "downsample the raster or process it in blocks"}

        # ---- embed on a registered grid
        n_r, n_c = -(-height // tile), -(-width // tile)
        windows = [(r, c) for r in range(n_r) for c in range(n_c)]
        dev = next(model.parameters()).device
        feats: np.ndarray | None = None
        try:
            for i in range(0, len(windows), max(1, batch_size)):
                chunk = windows[i:i + batch_size]
                batch = []
                for r, c in chunk:
                    y0, x0 = r * tile, c * tile
                    t = rgb[:, y0:y0 + tile, x0:x0 + tile]
                    ph, pw = tile - t.shape[1], tile - t.shape[2]
                    if ph or pw:   # replicate the edge; zero padding would be clustered as a class
                        t = np.pad(t, ((0, 0), (0, ph), (0, pw)), mode="edge")
                    batch.append((t - mean_a) / std_a)
                x = torch.from_numpy(np.stack(batch)).to(dev)
                with torch.no_grad():
                    tokens = model(pixel_values=x).last_hidden_state
                if tokens.shape[1] < per * per:
                    raise RuntimeError(f"backbone returned {tokens.shape[1]} tokens for a {per}x{per} grid")
                tokens = tokens[:, -per * per:]            # patch tokens are last (CLS/register tokens first)
                tokens = tokens.reshape(len(chunk), per, per, -1).float().cpu().numpy()
                if feats is None:
                    feats = np.zeros((n_r * per, n_c * per, tokens.shape[-1]), np.float32)
                for (r, c), tk in zip(chunk, tokens):
                    feats[r * per:(r + 1) * per, c * per:(c + 1) * per] = tk
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"embedding failed: {type(exc).__name__}: {exc}"}
        assert feats is not None
        feats = feats[:rows, :cols]

        # ---- cells that are entirely nodata are neither clustered nor labelled
        vpad = np.zeros((rows * patch, cols * patch), bool)
        vpad[:height, :width] = valid
        cell_valid = vpad.reshape(rows, patch, cols, patch).any(axis=(1, 3))
        flat = feats.reshape(rows * cols, -1)
        keep = cell_valid.reshape(-1)

        labels = np.full(rows * cols, NODATA_LABEL, np.uint8)
        x_valid = flat[keep]
        if clustering == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            k = int(min(n_classes, len(x_valid)))
            ids = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(x_valid)
            n_found = k
        else:
            try:
                import hdbscan
            except ImportError:
                return {"success": False, "error": "pip install hdbscan"}
            ids = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(x_valid)
            n_found = len(set(ids.tolist()) - {-1})
            ids = np.where(ids < 0, NODATA_LABEL, ids)     # noise is "unlabelled", not a class
        if n_found >= NODATA_LABEL:
            return {"success": False, "error": f"{n_found} clusters do not fit a uint8 label raster"}
        labels[keep] = ids.astype(np.uint8)

        grid = labels.reshape(rows, cols)
        full = np.repeat(np.repeat(grid, patch, axis=0), patch, axis=1)[:height, :width]
        full = np.where(valid, full, NODATA_LABEL).astype(np.uint8)

        # ---- write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_profile = {k: profile[k] for k in ("crs", "transform") if k in profile}
        out_profile.update(driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                           nodata=NODATA_LABEL, compress="lzw")
        from pygeovision.data._profile import writable_profile
        out_profile = writable_profile(out_profile)
        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(full[np.newaxis])
            dst.update_tags(source=f"FoundationModelLabeler_{self.model_name}", n_classes=str(n_found),
                            clustering=clustering, cell_px=str(patch), rgb_bands=str(tuple(bands)))

        return {
            "success": True,
            "output_path": str(output_path),
            "n_classes": n_found,
            "n_patches": int(keep.sum()),
            "cell_px": patch,
            "valid_fraction": float(valid.mean()),
            "model": self.model_name,
        }
