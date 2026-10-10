"""
Foundation Model Labeler.

Pseudo-labels for satellite tiles from foundation-model embeddings: embed the tile on a regular
grid, cluster the embeddings (or, for RemoteCLIP, match them to text prompts), and paint each
grid cell's label back onto the pixels it covers.

Supported models:
    - ``prithvi``      Prithvi-EO-1.0-100M, six HLS bands, one embedding per 16x16-pixel patch
    - ``prithvi_600``  Prithvi-EO-2.0-600M, six HLS bands, one embedding per 14x14-pixel patch
    - ``remote_clip``  RemoteCLIP ViT-L/14 (needs ``open_clip_torch``), one embedding per 224-px
      tile, optionally matched to ``text_prompts`` for zero-shot labels

What this labeller will **not** do is substitute another model when the one you asked for cannot
be loaded.  Earlier versions fell back to an ImageNet ViT for Prithvi and to OpenAI CLIP for
RemoteCLIP, and on any inference error used all-zero embeddings, which were then clustered and
reported as a successful labelling.  Now the tile fails with the reason.

``clay`` and ``satmae`` have no verified loader in this package and raise ``NotImplementedError``.

Example:
    >>> from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler
    >>> labeler = FoundationModelLabeler(model="prithvi", num_classes=8)
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# patch_size is the number of *source pixels* one embedding covers; image_size is the tile fed to the model.
_FOUNDATION_MODELS: dict[str, dict[str, Any]] = {
    "prithvi": {
        "backend": "prithvi",
        "variant": "prithvi_eo_1_0",
        "hub_id": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        "input_bands": 6,  # Blue, Green, Red, Narrow NIR, SWIR-1, SWIR-2 (HLS order)
        "patch_size": 16,
        "image_size": 224,
        "description": "NASA/IBM Prithvi-EO-1.0 100M (HLS imagery)",
    },
    "prithvi_600": {
        "backend": "prithvi",
        "variant": "prithvi_eo_2_0_600m",
        "hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "input_bands": 6,
        "patch_size": 14,
        "image_size": 224,
        "description": "NASA/IBM Prithvi-EO-2.0 600M (HLS imagery)",
    },
    "remote_clip": {
        "backend": "clip",
        "clip_name": "remoteclip-l14",
        "hub_id": "chendelong/RemoteCLIP",
        "input_bands": 3,
        "patch_size": 224,   # one embedding per tile
        "image_size": 224,
        "description": "RemoteCLIP ViT-L/14 vision-language model for remote sensing",
    },
}

_UNSUPPORTED: dict[str, str] = {
    "clay": "Clay has no verified loader in this package (its Hub repo is not a transformers model).",
    "satmae": "SatMAE has no verified loader in this package (no transformers implementation).",
}


@dataclass
class FoundationLabelConfig:
    """Configuration for the Foundation Model Labeler.

    Attributes:
        model: Foundation model name (key in _FOUNDATION_MODELS).
        task: ``'land_cover'`` (cluster the embeddings; with ``remote_clip`` + ``text_prompts``,
            match them to the prompts instead).
        text_prompts: Class text prompts for RemoteCLIP; label ``k`` is ``text_prompts[k]``.
        num_classes: Number of k-means clusters.
        device: Compute device.
        checkpoint_dir: Local model cache directory.
        batch_size: Tiles per forward pass.
        input_bands: 0-based band indices of the raster to feed to the model.
    """

    model: str = "prithvi"
    task: str = "land_cover"
    text_prompts: list[str] | None = None
    num_classes: int = 10
    cluster_method: str = "kmeans"
    device: str = "cpu"
    checkpoint_dir: Path | None = None
    batch_size: int = 16
    input_bands: list[int] | None = None


class FoundationModelLabeler(BaseLabeler):
    """Pseudo-label generator using geospatial foundation models.

    Workflow: (1) load the model lazily; (2) cut the tile into non-overlapping model-sized
    windows; (3) embed each window (Prithvi: every patch; RemoteCLIP: the window); (4) cluster
    (k-means on PCA-reduced, standardised embeddings) or match to text prompts; (5) paint every
    cell's label onto exactly the pixels it covers.

    Clusters are unnamed: cluster ``3`` means "tiles that look alike", not "forest".  Name them by
    inspecting them or by using ``remote_clip`` with ``text_prompts``.

    Args:
        model: ``'prithvi'``, ``'prithvi_600'`` or ``'remote_clip'``.
        task: only ``'land_cover'`` is implemented.
        text_prompts: RemoteCLIP class descriptions, e.g. ``["forest", "water", "urban"]``.
        num_classes: k-means clusters.
        device: Compute device. Auto-detected if None.
        input_bands: 0-based band indices to pass to the model (default: the first N).
        num_workers: Parallel workers.
        skip_existing: Skip tiles that already have labels.
    """

    def __init__(
        self,
        model: str = "prithvi",
        task: str = "land_cover",
        text_prompts: list[str] | None = None,
        num_classes: int = 10,
        device: str | None = None,
        input_bands: list[int] | None = None,
        num_workers: int = 1,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name=f"foundation_{model}",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if model in _UNSUPPORTED:
            raise NotImplementedError(f"model {model!r}: {_UNSUPPORTED[model]}")
        if model not in _FOUNDATION_MODELS:
            raise ValueError(
                f"model must be one of {list(_FOUNDATION_MODELS.keys())}, got {model!r}"
            )
        if task != "land_cover":
            raise NotImplementedError(
                f"task {task!r} is not implemented; only 'land_cover' (clustering / text matching) is.")
        if text_prompts and model != "remote_clip":
            raise ValueError("text_prompts are only used with model='remote_clip'")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        resolved_device = device or self._detect_device()
        self.config = FoundationLabelConfig(
            model=model,
            task=task,
            text_prompts=text_prompts,
            num_classes=num_classes,
            device=resolved_device,
            checkpoint_dir=Path.home() / ".pygeovision" / "cache" / "foundation_models",
            input_bands=input_bands,
        )
        self._model_info = _FOUNDATION_MODELS[model]
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'foundation'

    @property
    def supported_tasks(self) -> list:
        return ['segmentation', 'classification']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Generate pseudo-labels for a single satellite tile."""
        try:
            import rasterio
        except ImportError as exc:
            raise LabelingError(
                "foundation_labeler requires rasterio. Install: pip install rasterio"
            ) from exc

        try:
            self._ensure_model_loaded()

            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                transform = src.transform
                band_count = src.count

            image = self._load_bands(tile_path, band_count)
            embeddings, patch_grid, cell_px = self._extract_embeddings(image)

            if self.config.model == "remote_clip" and self.config.text_prompts:
                patch_labels = self._classify_with_text(embeddings)
            else:
                patch_labels = self._cluster_embeddings(embeddings)

            mask = self._upsample_patch_labels(
                patch_labels=patch_labels, patch_grid=patch_grid, cell_px=cell_px,
                height=height, width=width,
            )

            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_mask_geotiff(mask, output_path, meta)

            unique_classes = np.unique(mask)
            stats = {f"class_{c}": float(np.sum(mask == c)) / mask.size for c in unique_classes}
            metadata = {
                "model": self.config.model,
                "task": self.config.task,
                "num_classes": len(unique_classes),
                "embedding_dim": int(embeddings.shape[-1]),
                "n_embeddings": int(embeddings.shape[0]),
                "cell_px": cell_px,
                "device": self.config.device,
            }
            if self.config.text_prompts and self.config.model == "remote_clip":
                metadata["class_names"] = list(self.config.text_prompts)

            return LabelingResult(
                tile_id=str(tile_path),
                label_path=output_path,
                source=f"foundation_{self.config.model}",
                class_distribution=stats,
                metadata=metadata,
            )

        except Exception as exc:
            logger.error("FoundationModelLabeler failed for %s: %s", tile_path, exc)
            return LabelingResult(
                tile_id=str(tile_path),
                label_path=None,
                source=f"foundation_{self.config.model}",
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Load exactly the requested model, or raise ``LabelingError`` saying why not."""
        if self._model is not None:
            return
        info = self._model_info
        logger.info("Loading foundation model '%s' (%s)…", self.config.model, info["hub_id"])
        try:
            if info["backend"] == "prithvi":
                from pygeovision.models.foundation.prithvi import Prithvi
                self._model = Prithvi(info["variant"], device=self.config.device, num_frames=1).load()
            else:
                from pygeovision.models.vlm.clip import CLIPGeo
                self._model = CLIPGeo(model=info["clip_name"], device=self.config.device,
                                      cache_dir=str(self.config.checkpoint_dir))
                self._model._load()
        except Exception as exc:
            raise LabelingError(
                f"Failed to load foundation model '{self.config.model}': {exc}",
                details="No other model was substituted.",
            ) from exc
        logger.info("Foundation model '%s' loaded.", self.config.model)

    def _load_bands(self, tile_path: Path, band_count: int) -> np.ndarray:
        """Read the model's input bands as float32 ``(C, H, W)``.

        Prithvi: surface reflectance in [0, 1] (values above 10 are taken to be DN x 10000); the
        model standardises it itself.  RemoteCLIP: percentile-stretched to [0, 1], ignoring nodata.
        """
        import rasterio

        n_bands = self._model_info["input_bands"]
        if self.config.input_bands:
            if len(self.config.input_bands) != n_bands:
                raise ValueError(f"{self.config.model} needs {n_bands} input_bands, "
                                 f"got {len(self.config.input_bands)}")
            band_indices = [b + 1 for b in self.config.input_bands]
        else:
            if band_count < n_bands:
                raise ValueError(f"{self.config.model} needs {n_bands} bands, {tile_path} has {band_count}")
            band_indices = list(range(1, n_bands + 1))
        if max(band_indices) > band_count:
            raise ValueError(f"band index {max(band_indices) - 1} out of range for {band_count} bands")

        with rasterio.open(tile_path) as src:
            data = src.read(band_indices, masked=True).astype(np.float32)

        if self._model_info["backend"] == "prithvi":
            arr = data.filled(0.0)
            if float(arr.max()) > 10.0:
                arr = arr / 10000.0
            return np.clip(arr, 0.0, 1.0).astype(np.float32)

        out = np.zeros(data.shape, dtype=np.float32)
        for i in range(data.shape[0]):
            valid = data[i].compressed()
            if valid.size:
                p2, p98 = np.percentile(valid, (2, 98))
                out[i] = np.clip((data[i].filled(p2) - p2) / max(p98 - p2, 1e-6), 0, 1)
        return out

    def _extract_embeddings(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
        """Embed the image on a regular grid.

        Returns ``(embeddings (N, D), (rows, cols), cell_px)``: ``embeddings`` is row-major over a
        ``rows x cols`` grid whose cells are ``cell_px`` source pixels square, starting at the
        top-left pixel.  Every embedding is a real model output -- any failure propagates.
        """
        import torch

        info = self._model_info
        size, cell = int(info["image_size"]), int(info["patch_size"])
        _, H, W = image.shape
        n_r, n_c = -(-H // size), -(-W // size)
        per = size // cell                       # grid cells per tile side (1 for RemoteCLIP)

        windows, origins = [], []
        for r in range(n_r):
            for c in range(n_c):
                win = image[:, r * size:(r + 1) * size, c * size:(c + 1) * size]
                ph, pw = size - win.shape[1], size - win.shape[2]
                if ph or pw:                     # edge tile: replicate border instead of zero-filling
                    win = np.pad(win, ((0, 0), (0, ph), (0, pw)), mode="edge")
                windows.append(win)
                origins.append((r, c))

        grid = np.zeros((n_r * per, n_c * per, 0), dtype=np.float32)
        tiles_emb: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(windows), self.config.batch_size):
                batch = windows[i:i + self.config.batch_size]
                if info["backend"] == "prithvi":
                    x = torch.from_numpy(np.stack(batch)).to(self.config.device)
                    out = self._model._model(pixel_values=x)
                    gh, gw = out.grid
                    if (gh, gw) != (per, per):
                        raise RuntimeError(f"unexpected token grid {(gh, gw)}, expected {(per, per)}")
                    emb = out.frame_tokens[:, 0].cpu().numpy()                 # (B, per*per, D)
                else:
                    from PIL import Image
                    emb = np.stack([
                        self._model.embed_pil(Image.fromarray(
                            (w.transpose(1, 2, 0) * 255).astype(np.uint8)))
                        for w in batch])[:, None, :]                          # (B, 1, D)
                tiles_emb.append(emb)
        emb_all = np.concatenate(tiles_emb, axis=0)                           # (T, per*per, D)
        d = emb_all.shape[-1]

        grid = np.zeros((n_r * per, n_c * per, d), dtype=np.float32)
        for (r, c), e in zip(origins, emb_all):
            grid[r * per:(r + 1) * per, c * per:(c + 1) * per] = e.reshape(per, per, d)
        rows, cols = -(-H // cell), -(-W // cell)                              # cells touching real pixels
        grid = grid[:rows, :cols]
        return grid.reshape(-1, d), (rows, cols), cell

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings with k-means (PCA to <=64 dims, standardised first)."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        n_components = min(64, embeddings.shape[0] - 1, embeddings.shape[1])
        reduced = PCA(n_components=n_components).fit_transform(embeddings) if n_components > 1 else embeddings
        scaled = StandardScaler().fit_transform(reduced)

        n_clusters = min(self.config.num_classes, embeddings.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        return kmeans.fit_predict(scaled).astype(np.uint8)

    def _classify_with_text(self, embeddings: np.ndarray) -> np.ndarray:
        """Zero-shot labels: nearest text prompt (cosine similarity) for every embedding."""
        prompts = self.config.text_prompts
        if not prompts:
            raise ValueError("text_prompts are required for zero-shot classification")
        if len(prompts) > 255:
            raise ValueError("at most 255 text prompts (labels are uint8)")
        text = np.stack([self._model.embed_text(p) for p in prompts])           # (K, D), L2-normalised
        emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return np.argmax(emb @ text.T, axis=1).astype(np.uint8)

    @staticmethod
    def _upsample_patch_labels(
        patch_labels: np.ndarray,
        patch_grid: tuple[int, int],
        cell_px: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Paint each grid cell onto the ``cell_px`` x ``cell_px`` pixels it covers.

        Cell ``(i, j)`` covers pixel rows ``i*cell_px ...`` and columns ``j*cell_px ...``, so labels
        stay registered with the imagery (the old version stretched the grid over the image).
        """
        rows, cols = patch_grid
        grid = np.asarray(patch_labels)[:rows * cols].reshape(rows, cols).astype(np.uint8)
        full = np.repeat(np.repeat(grid, cell_px, axis=0), cell_px, axis=1)
        return full[:height, :width]
