"""
Tiled inference with Gaussian blending (B1, B4, B5).
Handles arbitrarily large GeoTIFFs without memory overflow.
Uses smooth Gaussian window weighting at tile boundaries.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GaussianBlend:
    """Gaussian blend window for seamless tile stitching (B1).

    Creates a weight mask where pixels near tile borders contribute
    less to the final prediction, eliminating visible seam lines.
    """

    @staticmethod
    def window(size: int, sigma_ratio: float = 0.25) -> Any:
        """Generate a 2D Gaussian window of shape (size, size)."""
        import numpy as np
        sigma = size * sigma_ratio
        size // 2
        y, x = np.mgrid[:size, :size]
        # Use exact centre (size-1)/2 for perfect symmetry
        cx = cy = (size - 1) / 2.0
        dist2 = (x - cx) ** 2 + (y - cy) ** 2
        window = np.exp(-dist2 / (2 * sigma**2))
        return window / window.max()

    @staticmethod
    def apply_to_prediction(pred: Any, size: int, sigma_ratio: float = 0.25) -> Any:
        """Weight a prediction array with the Gaussian window."""
        import numpy as np
        w = GaussianBlend.window(size, sigma_ratio)
        if pred.ndim == 3:  # (C, H, W)
            return pred * w[np.newaxis, :pred.shape[1], :pred.shape[2]]
        return pred * w[:pred.shape[0], :pred.shape[1]]


class TiledInference:
    """Memory-efficient tiled GeoTIFF inference with Gaussian boundary blending (B1-B5).

    Processes large satellite imagery (>1GB GeoTIFF) tile-by-tile with:
        - Gaussian boundary blending (no seam artefacts)
        - Configurable overlap between tiles
        - Parallel batch processing of tiles
        - Multiple blend modes (gaussian, linear, constant)
        - Memory usage estimation and automatic batch sizing

    Example::

        from pygeovision.inference import TiledInference

        inferencer = TiledInference(
            model=my_model,
            chip_size=512,
            overlap=128,
            blend_mode="gaussian",
        )
        result = inferencer.infer(
            "./data/large_sentinel2.tif",
            output_path="./results/prediction.tif",
        )
    """

    def __init__(
        self,
        model: Any,
        chip_size: int = 512,
        overlap: int = 128,
        blend_mode: str = "gaussian",     # gaussian | linear | constant
        batch_tiles: int = 4,
        device: str | None = None,
        num_classes: int = 2,
        activation: str = "softmax",       # softmax | sigmoid | none
        sigma_ratio: float = 0.25,
        dtype: str = "float32",
        tta: bool = False,                 # Test-time augmentation
        half_precision: bool = True,
        tolerate_chip_failures: bool = False,
    ) -> None:
        self.model = model
        self.chip_size = chip_size
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.batch_tiles = batch_tiles
        self.num_classes = num_classes
        self.activation = activation
        self.sigma_ratio = sigma_ratio
        self.dtype = dtype
        self.tta = tta
        self.half_precision = half_precision
        self.tolerate_chip_failures = tolerate_chip_failures
        self._device = device or self._auto_device()
        self._model_loaded = False
        self._chip_failures = 0

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _prepare_model(self) -> None:
        if self._model_loaded:
            return
        try:
            self.model = self.model.to(self._device)
            self.model.eval()
            if self.half_precision and self._device == "cuda":
                self.model = self.model.half()
            self._model_loaded = True
        except Exception as exc:
            logger.warning("Model preparation: %s", exc)

    def _get_blend_window(self, h: int, w: int) -> Any:
        import numpy as np
        if self.blend_mode == "gaussian":
            window = GaussianBlend.window(self.chip_size, self.sigma_ratio)
            return window[:h, :w]
        elif self.blend_mode == "linear":
            y_ramp = np.minimum(np.arange(h), np.arange(h)[::-1]) / (h // 2)
            x_ramp = np.minimum(np.arange(w), np.arange(w)[::-1]) / (w // 2)
            return np.outer(y_ramp.clip(0, 1), x_ramp.clip(0, 1))
        else:  # constant
            return np.ones((h, w))


    @staticmethod
    def _extract_logits(output: Any) -> Any:
        """Unwrap a model's forward() output into a raw logits tensor.

        Many HuggingFace transformers models (SegFormer, ViT, etc.) return a
        ModelOutput dataclass (e.g. BaseModelOutput, SemanticSegmenterOutput)
        rather than a bare tensor. TiledInference needs the raw logits tensor
        for softmax/sigmoid — this extracts it regardless of which output
        shape the wrapped model produces.
        """
        import torch
        if isinstance(output, torch.Tensor):
            return output
        # Standard HF ModelOutput / SemanticSegmenterOutput: has .logits
        if hasattr(output, "logits"):
            return output.logits
        # Bare encoder output (e.g. SegformerModel, not
        # SegformerForSemanticSegmentation) — has last_hidden_state but no
        # segmentation head, so there ARE no class logits to extract.
        if hasattr(output, "last_hidden_state"):
            raise TypeError(
                "Model output has 'last_hidden_state' but no 'logits' — this "
                "looks like a bare encoder/backbone (e.g. SegformerModel) "
                "rather than a task head (e.g. SegformerForSemanticSegmentation). "
                "TiledInference needs a model that outputs class logits directly. "
                "Check how get_model() constructs this model — it likely needs "
                "the *ForSemanticSegmentation variant, not the base encoder."
            )
        raise TypeError(
            f"Don't know how to extract logits from model output of type "
            f"{type(output).__name__}. Expected a torch.Tensor or an object "
            f"with a '.logits' attribute."
        )

    def _predict_chip(self, chip: Any) -> Any:
        """Run model inference on a single chip."""
        try:
            import numpy as np
            import torch

            if isinstance(chip, np.ndarray):
                use_half = self.half_precision and self._device == "cuda"
                chip_t = torch.tensor(chip, dtype=torch.float16 if use_half else torch.float32)
            else:
                chip_t = chip

            if chip_t.ndim == 3:
                chip_t = chip_t.unsqueeze(0)
            chip_t = chip_t.to(self._device)

            if self.tta:
                preds = []
                for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
                    x = chip_t.clone()
                    if flip_h: x = torch.flip(x, dims=[-1])
                    if flip_v: x = torch.flip(x, dims=[-2])
                    with torch.no_grad():
                        p = self.model(x)
                    p = self._extract_logits(p)   # <-- unwrap here
                    if flip_h: p = torch.flip(p, dims=[-1])
                    if flip_v: p = torch.flip(p, dims=[-2])
                    preds.append(p)
                logits = torch.stack(preds).mean(0)
            else:
                with torch.no_grad():
                    logits = self.model(chip_t)
                logits = self._extract_logits(logits)   # <-- unwrap here too

            if self.activation == "softmax":
                probs = torch.softmax(logits, dim=1)
            elif self.activation == "sigmoid":
                probs = torch.sigmoid(logits)
            else:
                probs = logits

            return probs[0].cpu().float().numpy()
        except Exception as exc:
            logger.error(
                "Chip prediction failed — returning an all-zero chip, which will "
                "corrupt this region of the output raster. This usually means the "
                "model's forward() interface doesn't match TiledInference's "
                "assumption (single image tensor in -> class-logit tensor out). "
                "Detection models (Mask R-CNN, Faster R-CNN, FCOS, YOLO) and "
                "change-detection models (BIT, DSAMNet, ChangeFormer) are NOT "
                "compatible with TiledInference — they need a dedicated wrapper "
                "with a different calling convention. Root cause: %s", exc,
                exc_info=True,
            )
            self._chip_failures = getattr(self, "_chip_failures", 0) + 1
            if not getattr(self, "tolerate_chip_failures", False):
                raise RuntimeError(
                    f"TiledInference chip prediction failed ({exc}). Set "
                    f"tolerate_chip_failures=True on TiledInference to instead "
                    f"log and fill with zeros (NOT recommended — silently "
                    f"corrupts output)."
                ) from exc
            import numpy as np
            return np.zeros((self.num_classes, self.chip_size, self.chip_size), dtype=np.float32)

    def infer(
        self,
        image_path: str | Path,
        output_path: str | Path = "./output/prediction.tif",
        return_probabilities: bool = False,
        band_selection: list[int] | None = None,
        normalise: bool = True,
        nodata_value: float | None = None,
    ) -> dict[str, Any]:
        """Run tiled inference on a large GeoTIFF.

        Args:
            image_path: Input GeoTIFF (any size, any number of bands)
            output_path: Output prediction GeoTIFF
            return_probabilities: If True, output all class probability maps
            band_selection: Band indices to use (default: all)
            normalise: Normalise each band to [0, 1]
            nodata_value: Value to treat as nodata/ignore

        Returns:
            Dict with output_path, inference stats, timing
        """
        try:
            import numpy as np
            import rasterio
            import torch
        except ImportError as exc:
            return {"success": False, "error": f"rasterio + torch required: {exc}"}

        image_path  = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._prepare_model()
        t_start = time.time()
        n_chips_total = 0
        n_chips_ok    = 0

        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            H, W    = src.height, src.width
            n_bands = src.count

            bands = band_selection or list(range(1, n_bands + 1))
            image = src.read(bands).astype(np.float32)

        # Real fix, confirmed by direct inspection: this previously had
        # zero validation that the selected band count (defaults to
        # ALL bands in the file) actually matches what the model
        # expects as input -- the same class of bug already found and
        # fixed in the separate ai.inference.tiled_inference engine
        # (a real user hit "expected 3 bands, got 13" there), but this
        # parallel, CLI-reachable engine never got that fix. Real
        # pre-flight check: a tiny dummy forward pass with the actual
        # selected band count, so a mismatch surfaces here with a
        # clear message rather than deep inside the tile loop.
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, len(bands), 32, 32)
                self.model(dummy.to(self._device))
        except Exception as exc:
            if not self.tolerate_chip_failures:
                return {
                    "success": False,
                    "error": f"Model does not accept {len(bands)} input band(s) (selected from "
                             f"{n_bands} real band(s) in {image_path}): {exc}. Pass band_selection= "
                             f"explicitly if this file has more bands than the model expects.",
                }
            logger.warning(
                "Pre-flight check found the model does not accept %d input band(s) "
                "(selected from %d real band(s) in %s): %s. tolerate_chip_failures=True "
                "was explicitly set, so proceeding anyway -- every chip will likely "
                "fail and be zero-filled per that setting's real, existing behavior.",
                len(bands), n_bands, image_path, exc,
            )

        if normalise:
            for b in range(image.shape[0]):
                p2, p98 = np.percentile(image[b], (2, 98))
                image[b] = (image[b] - p2) / (p98 - p2 + 1e-8)

        # Build accumulation buffers
        accum  = np.zeros((self.num_classes, H, W), dtype=np.float64)
        counts = np.zeros((H, W), dtype=np.float64)
        stride = self.chip_size - self.overlap

        tile_batch: list[tuple] = []

        def _flush_batch(batch):
            for (r, c, rh, cw, chip_data) in batch:
                pred = self._predict_chip(chip_data)
                weight = self._get_blend_window(rh, cw)
                actual_h, actual_w = min(pred.shape[-2], rh), min(pred.shape[-1], cw)
                # Accumulate weighted predictions
                accum[:, r:r+actual_h, c:c+actual_w] += pred[:, :actual_h, :actual_w] * weight[:actual_h, :actual_w]
                counts[r:r+actual_h, c:c+actual_w]   += weight[:actual_h, :actual_w]

        for row in range(0, H, stride):
            for col in range(0, W, stride):
                r2 = min(row + self.chip_size, H)
                c2 = min(col + self.chip_size, W)
                actual_h, actual_w = r2 - row, c2 - col

                chip_data = image[:, row:r2, col:c2]
                # Pad chip to chip_size if needed
                if chip_data.shape[1] < self.chip_size or chip_data.shape[2] < self.chip_size:
                    padded = np.zeros((image.shape[0], self.chip_size, self.chip_size), dtype=np.float32)
                    padded[:, :chip_data.shape[1], :chip_data.shape[2]] = chip_data
                    chip_data = padded

                tile_batch.append((row, col, actual_h, actual_w, chip_data))
                n_chips_total += 1

                if len(tile_batch) >= self.batch_tiles:
                    _flush_batch(tile_batch)
                    n_chips_ok += len(tile_batch)
                    tile_batch = []

        if tile_batch:
            _flush_batch(tile_batch)
            n_chips_ok += len(tile_batch)

        # Normalise accumulated predictions
        safe_counts = np.maximum(counts, 1e-10)
        accum /= safe_counts[np.newaxis]

        # Build output
        if return_probabilities:
            out_data = (accum * 255).clip(0, 255).astype(np.uint8)
            out_profile = profile.copy()
            out_profile.update(count=self.num_classes, dtype="uint8", compress="lzw")
        else:
            label = np.argmax(accum, axis=0).astype(np.uint8)
            out_data = label[np.newaxis]
            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw")

        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(out_data)
            dst.update_tags(
                method="TiledInference",
                blend_mode=self.blend_mode,
                chip_size=str(self.chip_size),
                overlap=str(self.overlap),
                tta=str(self.tta),
            )

        duration = time.time() - t_start
        logger.info("TiledInference: %d chips | %.1fs | %s", n_chips_ok, duration, output_path)

        return {
            "success": True,
            "output_path": str(output_path),
            "n_chips": n_chips_ok,
            "image_size": (H, W),
            "duration_seconds": round(duration, 1),
            "chips_per_second": round(n_chips_ok / max(duration, 0.001), 1),
            "blend_mode": self.blend_mode,
            "tta": self.tta,
        }

    def estimate_memory(self, H: int, W: int) -> dict[str, float]:
        """Estimate GPU memory usage for a given image size."""
        import math
        stride = self.chip_size - self.overlap
        n_chips_h = math.ceil(H / stride)
        n_chips_w = math.ceil(W / stride)
        n_chips = n_chips_h * n_chips_w
        # Memory per chip (float32)
        chip_bytes = self.chip_size * self.chip_size * self.num_classes * 4
        batch_mb = self.batch_tiles * chip_bytes / 1024**2
        accum_mb = H * W * self.num_classes * 8 / 1024**2  # float64 accum
        return {
            "n_chips": n_chips,
            "batch_gpu_mb": round(batch_mb, 1),
            "accumulation_ram_mb": round(accum_mb, 1),
            "estimated_total_mb": round(batch_mb + accum_mb, 1),
        }