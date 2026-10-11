"""Streaming inference for ultra-large GeoTIFFs (B3, B4) — never loads full image."""
from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StreamingInference:
    """Lower-memory variant of :class:`TiledInference` (one chip per model call).

    :meth:`stream_chips` reads one window at a time from disk. :meth:`infer`, however,
    delegates to :meth:`TiledInference.infer` with ``batch_tiles=1``: that method reads the
    whole scene into RAM and keeps full-size accumulators, so memory still grows with the
    scene (about ``bands * 4 + classes * 8 + 8`` bytes per pixel). ``batch_tiles=1`` only
    lowers the *model's* memory use, not the scene's. For scenes larger than RAM, cut the
    scene into pieces first (``pygeovision preprocess clip``), predict each piece, and mosaic.

    Note that :meth:`stream_chips` min-max scales each chip on its own, which is a different
    scaling from the per-scene percentile stretch of :meth:`infer`.
    """

    def __init__(self, model: Any, chip_size: int = 1024,
                 overlap: int = 128, num_classes: int = 2,
                 device: str | None = None) -> None:
        self.model = model
        self.chip_size = chip_size
        self.overlap = overlap
        self.num_classes = num_classes
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

    @staticmethod
    def _has_cuda():
        try:
            import torch; return torch.cuda.is_available()
        except ImportError: return False

    def stream_chips(self, image_path: str) -> Generator[dict, None, None]:
        """Generator that yields image chips one at a time (never loads full image)."""
        try:
            import numpy as np
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            raise ImportError("rasterio required")

        with rasterio.open(image_path) as src:
            H, W = src.height, src.width
            stride = self.chip_size - self.overlap

            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2 = min(row + self.chip_size, H)
                    c2 = min(col + self.chip_size, W)
                    window = Window(col, row, c2 - col, r2 - row)
                    chip = src.read(window=window).astype(np.float32)
                    # Normalise
                    for b in range(chip.shape[0]):
                        mn, mx = chip[b].min(), chip[b].max()
                        chip[b] = (chip[b] - mn) / (mx - mn + 1e-8)
                    yield {"chip": chip, "row": row, "col": col,
                            "height": r2 - row, "width": c2 - col,
                            "window": (col, row, c2 - col, r2 - row)}

    def infer(self, image_path: str | Path,
               output_path: str | Path, normalise: bool = True) -> dict[str, Any]:
        """Predict a GeoTIFF one chip per model call (see the class note on memory)."""
        from pygeovision.inference.tiled import TiledInference
        # Delegate to TiledInference with batch_tiles=1 (minimal memory)
        inf = TiledInference(
            model=self.model, chip_size=self.chip_size,
            overlap=self.overlap, num_classes=self.num_classes,
            device=self.device, batch_tiles=1,
        )
        return inf.infer(image_path, output_path, normalise=normalise)


class EnsembleInference:
    """Combine the class probabilities of several models, then take the most likely class.

    ``weights`` are relative importances (default equal). ``fusion="mean"`` averages the
    weighted probabilities, ``"max"`` takes the per-class maximum; anything else raises
    ``ValueError``. The probabilities come from 8-bit rasters (steps of 1/255).
    """

    def __init__(self, models: list[Any], weights: list[float] | None = None,
                 fusion: str = "mean", num_classes: int = 2,
                 device: str | None = None, normalise: bool = True) -> None:
        if fusion not in ("mean", "max"):
            raise ValueError(f"fusion must be 'mean' or 'max', got {fusion!r}")
        if not models:
            raise ValueError("EnsembleInference needs at least one model")
        if weights is not None and len(weights) != len(models):
            raise ValueError(f"{len(weights)} weights for {len(models)} models")
        self.normalise = normalise
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.fusion = fusion
        self.num_classes = num_classes
        self.device = device or ("cuda" if StreamingInference._has_cuda() else "cpu")

    def infer(self, image_path: str | Path,
               output_path: str | Path) -> dict[str, Any]:
        import numpy as np
        import rasterio

        from pygeovision.inference.tiled import TiledInference

        all_probs = []
        for model, weight in zip(self.models, self.weights):
            inf = TiledInference(model=model, num_classes=self.num_classes, device=self.device)
            tmp = str(output_path) + f".tmp{len(all_probs)}.tif"
            try:
                res = inf.infer(image_path, tmp, return_probabilities=True, normalise=self.normalise)
                if res.get("success") is False:
                    return {"success": False, "error": res.get("error"), "n_models": len(self.models)}
                with rasterio.open(tmp) as src:
                    probs = src.read().astype(np.float32) / 255.0 * weight
                    all_probs.append(probs)
                    if len(all_probs) == 1:
                        profile = src.profile.copy()
            finally:
                Path(tmp).unlink(missing_ok=True)       # do not leave one temporary raster per model behind

        if self.fusion == "mean":
            fused = np.stack(all_probs).mean(0)
        elif self.fusion == "max":
            fused = np.stack(all_probs).max(0)

        label = np.argmax(fused, axis=0).astype(np.uint8)
        no_data = fused.max(axis=0) <= 0           # pixels the models were not asked to predict
        label[no_data] = 255
        profile.update(count=1, dtype="uint8", nodata=255 if no_data.any() else None)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from pygeovision.data._profile import writable_profile
        profile = writable_profile(profile)
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(label[np.newaxis])

        return {"success": True, "output_path": str(output_path), "n_models": len(self.models)}
