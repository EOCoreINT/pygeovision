"""
Few-shot learning for geospatial models (D3).
Classify new land cover classes with only 1-10 labelled examples.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FewShotLearner:
    """Prototype-based few-shot classification of image chips with a frozen foundation backbone.

    A handful of labelled example images per class (the *support set*) are turned into feature
    vectors by a frozen DINOv2 backbone; each class is represented by the **mean** of its vectors (its
    prototype); a new image gets the class of the nearest prototype. No weights are trained.
    Only ``method="prototypical"`` exists; MAML and matching networks are not implemented.

    The backbone sees an 8-bit RGB picture made from three bands of each file
    (``bands``, 1-based, default 1-3) stretched between the 2nd and 98th percentile of every image
    (``stretch``: one shared pair of percentiles, or one per band), or, with ``value_range=(low, high)``,
    scaled by one fixed range so that dim and bright images stay different.
    Pretrained ImageNet-style backbones are not made for multispectral data, so choose
    the three bands that look natural (for Sentinel-2 files written B2, B3, B4 pass ``bands=(3, 2, 1)``).

    Example::

        learner = FewShotLearner(backbone="dinov2-base")
        learner.fit_support({"mangrove": ["m1.tif", "m2.tif"], "urban": ["u1.tif", "u2.tif"]})
        preds = learner.predict(["query1.tif", "query2.tif"])
        learner.predict_geotiff("scene.tif", "classes.tif")
    """

    METHODS = ("prototypical",)
    METRICS = ("euclidean", "cosine")
    STRETCHES = ("joint", "per_band")

    # Short names for the DINOv2 family; anything else is passed to Hugging Face as it is
    # (a model id or a local directory). The backbone must return ``last_hidden_state`` with a
    # class token first, as DINOv2 and ViT models do.
    _HF_MODEL_IDS: dict[str, str] = {
        "dinov2-small":  "facebook/dinov2-small",
        "dinov2-base":   "facebook/dinov2-base",
        "dinov2-large":  "facebook/dinov2-large",
        "dinov2-giant":  "facebook/dinov2-giant",
    }

    def __init__(
        self,
        backbone: str = "dinov2-base",
        method: str = "prototypical",
        device: str | None = None,
        metric: str = "euclidean",
        temperature: float | None = None,
        bands: tuple[int, int, int] = (1, 2, 3),
        stretch: str = "joint",
        value_range: tuple[float, float] | None = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}; {method!r} is not implemented")
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {list(self.METRICS)}; got {metric!r}")
        if stretch not in self.STRETCHES:
            raise ValueError(f"stretch must be one of {list(self.STRETCHES)}; got {stretch!r}")
        if len(bands) != 3 or min(bands) < 1:
            raise ValueError(f"bands must be three 1-based band numbers; got {bands!r}")
        self.backbone = backbone
        self.method = method
        self.metric = metric
        # Distances become scores with softmax(-distance / temperature). Cosine distances are small
        # (0 to 2), so they need a small fixed temperature. Euclidean distances between backbone features
        # are tens of units, where a fixed temperature of 1 makes every score 0.0 or 1.0; by default the
        # temperature is therefore a quarter of the median distance between class prototypes (set in
        # fit_support), so a query as far from its class as the classes are from each other scores ~0.5.
        self._user_temperature = float(temperature) if temperature else None
        self.temperature = self._user_temperature or (1.0 if metric == "euclidean" else 0.05)
        if value_range is not None and not (len(value_range) == 2 and value_range[0] < value_range[1]):
            raise ValueError(f"value_range must be (low, high) with low < high; got {value_range!r}")
        self.bands = tuple(int(b) for b in bands)
        self.stretch = stretch
        self.value_range = tuple(float(v) for v in value_range) if value_range else None
        self.device = device or self._auto_device()
        self._feature_extractor = None
        self._processor = None
        self._prototypes: dict[str, Any] = {}
        self._class_names: list[str] = []

    @property
    def model_id(self) -> str:
        """Hugging Face model id (or path) for the current backbone."""
        return self._HF_MODEL_IDS.get(self.backbone, self.backbone)

    @property
    def class_names(self) -> list[str]:
        """Class names in output-code order (alphabetical): code ``i`` in a GeoTIFF is ``class_names[i]``."""
        return list(self._class_names)

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _load_backbone(self) -> None:
        if self._feature_extractor is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError("transformers + torch required: pip install transformers torch")
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._feature_extractor = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        except OSError as exc:
            raise RuntimeError(
                f"could not load backbone {self.model_id!r}: {exc}. It is downloaded from Hugging Face on "
                f"first use (needs a network); pass a local model directory as `backbone` to work offline."
            ) from exc
        logger.info("Few-shot backbone loaded: %s", self.model_id)

    def _read_rgb(self, path: str) -> Any:
        """8-bit RGB array ``(H, W, 3)`` from ``self.bands`` of a raster file."""
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            if src.count == 0:
                raise ValueError("file has no bands")
            wanted = [b for b in self.bands if b <= src.count]
            if not wanted:
                wanted = [1]
            data = src.read(wanted).astype(np.float64)
            nodata = src.nodata
        return self._to_rgb(data, nodata)

    def _to_rgb(self, data: Any, nodata: float | None = None) -> Any:
        """Scale to 8 bit between the 2nd and 98th percentile of the valid (finite, non-nodata) pixels.

        ``stretch="joint"`` (default) uses one pair of percentiles for all three bands, so the colour
        balance between bands survives (water stays dark blue, vegetation green); ``"per_band"``
        stretches each band on its own, which removes that balance and also any absolute colour.
        Either way brightness is relative to the image: a dim uniform scene and a bright one look alike.
        With ``value_range=(low, high)`` every image and chip is instead scaled linearly from that range, so
        brightness differences between images are kept.
        """
        import numpy as np
        data = np.array(data, dtype=np.float64, copy=True)
        ok = np.isfinite(data)
        if nodata is not None:
            ok &= data != nodata
        out = np.zeros_like(data)
        if self.value_range is not None:                  # the same scale for every image and chip
            lo, hi = self.value_range
            out = np.where(ok, np.clip((np.where(ok, data, lo) - lo) / (hi - lo) * 255, 0, 255), 0)
            while out.shape[0] < 3:
                out = np.concatenate([out, out[-1:]], axis=0)
            return out[:3].transpose(1, 2, 0).astype(np.uint8)
        groups = [list(range(data.shape[0]))] if self.stretch == "joint" else [[b] for b in range(data.shape[0])]
        for g in groups:
            valid = ok[g]
            if not valid.any():
                continue
            p2, p98 = np.percentile(data[g][valid], (2, 98))
            filled = np.where(ok[g], data[g], p2)
            out[g] = np.clip((filled - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        while out.shape[0] < 3:                           # one or two bands: repeat the last
            out = np.concatenate([out, out[-1:]], axis=0)
        return out[:3].transpose(1, 2, 0).astype(np.uint8)

    def _embed(self, rgb_images: list[Any], batch_size: int = 16) -> Any:
        """Class-token feature vectors ``(n, d)`` for a list of ``(H, W, 3)`` uint8 arrays."""
        import numpy as np
        import torch
        from PIL import Image

        self._load_backbone()
        feats = []
        for i in range(0, len(rgb_images), batch_size):
            pil = [Image.fromarray(a) for a in rgb_images[i:i + batch_size]]
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._feature_extractor(**inputs)
            feats.append(out.last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 0))

    def _extract_features(self, image_paths: list[str]) -> tuple[list[Any], list[str | None]]:
        """Feature vector per path, or ``None`` with the reason when that file could not be read."""
        rgbs: list[Any] = []
        errors: list[str | None] = []
        for path in image_paths:
            try:
                rgbs.append(self._read_rgb(path))
                errors.append(None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Feature extraction failed for %s: %s", path, exc)
                rgbs.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")
        good = [a for a in rgbs if a is not None]
        embedded = iter(self._embed(good)) if good else iter(())
        return [next(embedded) if a is not None else None for a in rgbs], errors

    def fit_support(self, support: dict[str, list[str]]) -> FewShotLearner:
        """Build one prototype per class from its example images.

        Args:
            support: ``{class_name: [image paths]}`` with at least two classes and, for each class,
                at least one image that can be read. Unreadable images are skipped with a warning.
                There is no upper limit, but the point of the method is 1-10 per class.

        Raises:
            ValueError: fewer than two classes, an empty class, or a class with no readable image.
        """
        import numpy as np

        if not isinstance(support, dict) or len(support) < 2:
            raise ValueError("support must be a dict with at least two classes {name: [paths]}")
        prototypes: dict[str, Any] = {}
        for class_name, paths in support.items():
            if isinstance(paths, (str, bytes)) or not len(paths):
                raise ValueError(f"class {class_name!r}: give a non-empty list of image paths")
            feats, errors = self._extract_features(list(paths))
            usable = [f for f in feats if f is not None]
            if not usable:
                raise ValueError(f"class {class_name!r}: none of its {len(paths)} images could be read "
                                 f"(first error: {next(e for e in errors if e)})")
            if len(usable) < len(paths):
                logger.warning("class %r: using %d of %d images", class_name, len(usable), len(paths))
            prototypes[class_name] = np.mean(usable, axis=0)
            logger.info("Prototype built: %s (%d examples)", class_name, len(usable))
        self._prototypes = prototypes
        self._class_names = sorted(prototypes)
        if self._user_temperature is None and self.metric == "euclidean":
            protos = [prototypes[n] for n in self._class_names]
            gaps = [float(np.linalg.norm(a - b)) for i, a in enumerate(protos) for b in protos[i + 1:]]
            median_gap = float(np.median(gaps))
            self.temperature = median_gap / 4.0 if median_gap > 1e-9 else 1.0
        return self

    def _scores(self, feat: Any) -> tuple[dict[str, float], dict[str, float]]:
        """``(distances, probabilities)`` of one feature vector against every prototype."""
        import numpy as np
        names = self._class_names
        if self.metric == "cosine":
            f = feat / (np.linalg.norm(feat) + 1e-12)
            dist = np.array([1.0 - float(f @ (self._prototypes[n] / (np.linalg.norm(self._prototypes[n]) + 1e-12)))
                             for n in names])
        else:
            dist = np.array([float(np.linalg.norm(feat - self._prototypes[n])) for n in names])
        logits = -dist / self.temperature
        e = np.exp(logits - logits.max())
        return dict(zip(names, dist.tolist())), dict(zip(names, (e / e.sum()).tolist()))

    def predict(
        self,
        query_paths: list[str],
        return_distances: bool = False,
    ) -> list[dict[str, Any]]:
        """Classify images by their nearest prototype.

        Returns one dict per query: ``{"class", "confidence", "probabilities"}`` (plus ``"distances"``
        with ``return_distances``). ``confidence`` is the softmax score of the winning class among the
        classes **you gave**: a relative score, not a calibrated probability, and there is no "none of
        these" option, so an image of an unseen kind still gets the nearest class. A file that cannot be
        read gives ``{"class": None, "confidence": 0.0, "probabilities": {}, "error": ...}``.
        """
        if not self._prototypes:
            raise RuntimeError("Call fit_support() before predict()")

        feats, errors = self._extract_features(list(query_paths))
        results = []
        for feat, err in zip(feats, errors):
            if feat is None:
                results.append({"class": None, "confidence": 0.0, "probabilities": {}, "error": err})
                continue
            distances, probs = self._scores(feat)
            best = min(distances, key=distances.get)
            result = {
                "class": best,
                "confidence": round(probs[best], 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            }
            if return_distances:
                result["distances"] = {k: round(v, 4) for k, v in distances.items()}
            results.append(result)
        return results

    def predict_geotiff(
        self,
        image_path: str,
        output_path: str,
        chip_size: int = 224,
        stride: int | None = None,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Classify a GeoTIFF chip by chip and write a class map.

        Chips of ``chip_size`` pixels are placed every ``stride`` pixels (default half a chip; the last row
        and column of chips are moved back so every chip is full-size; an image smaller than a chip is one
        chip). Each chip is classified as a whole picture, and every pixel receives the confidence-weighted
        votes of the chips covering it, so the map is coarse: object edges are as sharp as the chip
        spacing. Chips that contain only nodata or NaN are skipped; pixels no chip covered are 255 (nodata).

        The output is one ``uint8`` band on the input grid. Code ``i`` is ``class_names[i]`` (alphabetical);
        the same mapping is in the returned ``class_map`` and the file's ``class_map`` tag.
        Returns ``{"success", "output_path", "n_classes", "class_map", "n_chips", "n_skipped_chips"}`` or
        ``{"success": False, "error": ...}``.
        """
        if not self._prototypes:
            return {"success": False, "error": "Call fit_support() before predict_geotiff()"}
        try:
            import numpy as np
            import rasterio
            from rasterio.windows import Window

            stride = stride or max(chip_size // 2, 1)
            if chip_size < 1 or stride < 1:
                return {"success": False, "error": "chip_size and stride must be positive"}

            with rasterio.open(image_path) as src:
                H, W = src.height, src.width
                profile = src.profile.copy()
                nodata = src.nodata
                wanted = [b for b in self.bands if b <= src.count] or [1]

                def _starts(n: int) -> list[int]:
                    if n <= chip_size:
                        return [0]
                    s = list(range(0, n - chip_size + 1, stride))
                    if s[-1] + chip_size < n:
                        s.append(n - chip_size)
                    return s

                names = self._class_names
                accum = np.zeros((len(names), H, W), dtype=np.float32)
                counts = np.zeros((H, W), dtype=np.float32)
                pending: list[tuple[int, int, int, int, Any]] = []
                n_chips = n_skipped = 0

                def _flush() -> None:
                    nonlocal pending
                    if not pending:
                        return
                    embedded = self._embed([p[4] for p in pending], batch_size=batch_size)
                    for (r0, r1, c0, c1, _), feat in zip(pending, embedded):
                        _, probs = self._scores(feat)
                        best = max(probs, key=probs.get)
                        accum[names.index(best), r0:r1, c0:c1] += probs[best]
                        counts[r0:r1, c0:c1] += 1.0
                    pending = []

                for r0 in _starts(H):
                    for c0 in _starts(W):
                        r1, c1 = min(r0 + chip_size, H), min(c0 + chip_size, W)
                        data = src.read(wanted, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
                        valid = np.isfinite(data)
                        if nodata is not None:
                            valid &= data != nodata
                        if not valid.any():
                            n_skipped += 1
                            continue
                        pending.append((r0, r1, c0, c1, self._to_rgb(data, nodata)))
                        n_chips += 1
                        if len(pending) >= batch_size:
                            _flush()
                _flush()

            label = np.full((H, W), 255, dtype=np.uint8)
            covered = counts > 0
            label[covered] = np.argmax(accum[:, covered], axis=0).astype(np.uint8)

            import pathlib
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw", nodata=255)
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                out_profile.pop(_k, None)
            class_map = {str(i): name for i, name in enumerate(names)}
            with rasterio.open(output_path, "w", **out_profile) as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(method="FewShotLearner", class_map=json.dumps(class_map))

            return {"success": True, "output_path": output_path, "n_classes": len(names),
                    "class_map": class_map, "n_chips": n_chips, "n_skipped_chips": n_skipped}
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
