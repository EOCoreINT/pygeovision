"""
SAM Auto-Labeler (E2) — Segment Anything Model for automated label generation.
Generates segmentation masks without any manual annotation.
Fully native — uses HuggingFace transformers directly.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _tile_origins(size: int, chip: int, overlap: int) -> list[int]:
    """Start offsets of overlapping tiles covering ``size`` pixels.

    The last tile ends exactly at ``size``; no tile is emitted that lies wholly inside the
    previous tile's footprint (the old ``range(0, size, stride)`` did, which processed a
    useless sliver when ``size`` was a multiple of ``chip``).
    """
    stride = chip - overlap
    starts = [0]
    while starts[-1] + chip < size:
        starts.append(starts[-1] + stride)
    return starts


def _pixel_area_m2(transform: Any, crs: Any, height: int) -> float:
    """Ground area of one pixel in square metres.

    For a projected CRS this is ``|a*e|`` (assumed to be in metres).  For a geographic CRS the
    pixel size is in degrees, so the old ``|a*e|`` gave an "area in m2" that was really deg^2
    (about 1e-10 of the truth) and made every ``min_area_m2`` filter drop nothing.
    """
    import math

    if crs is not None and getattr(crs, "is_geographic", False):
        lat = transform.f + transform.e * (height / 2.0)
        m_lat = 111_320.0
        m_lon = 111_320.0 * math.cos(math.radians(lat))
        return abs(transform.a) * m_lon * abs(transform.e) * m_lat
    return abs(transform.a * transform.e)


def _to_numpy_bool(seg: Any) -> Any:
    import numpy as np

    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    return np.asarray(seg).astype(bool)


class SAMAutoLabeler:
    """Automated label generation using Segment Anything Model (SAM/SAM2).

    Zero-shot segmentation -- generates instance masks from satellite imagery without any
    training data.  Works through Hugging Face ``transformers`` (``AutoModelForMaskGeneration``),
    so both the SAM and SAM2 families load.

    Supports:
        - Automatic mask generation (grid of point prompts, tiled over large scenes)
        - GroundedSAM (text-prompt driven; needs the ``groundingdino`` package)
        - Post-filtering by area, predicted IoU and stability

    Masks are held as bounding-box crops, not full-scene arrays, so memory grows with the
    size of the objects rather than with ``n_masks * scene_size``.

    Example::

        labeler = SAMAutoLabeler(model="sam-vit-base")
        result = labeler.auto_label(
            image_path="./data/aerial.tif",
            output_path="./labels/sam_masks.tif",
            points_per_side=32,
            min_area_m2=50,
        )
    """

    HF_MODELS = {
        "sam-vit-huge":   "facebook/sam-vit-huge",
        "sam-vit-large":  "facebook/sam-vit-large",
        "sam-vit-base":   "facebook/sam-vit-base",
        "sam2-hiera-l":   "facebook/sam2-hiera-large-hf",
        "sam2-hiera-b":   "facebook/sam2-hiera-base-plus-hf",
    }
    # Names that used to be accepted but point at repositories that do not exist.
    _REMOVED = {
        "geosam-vit-h": "'wangyi111/GeoSAM' is not a Hugging Face repository. Use a SAM name from "
                        "HF_MODELS, or SamGeoLabeler (segment-geospatial) for remote-sensing SAM.",
    }

    def __init__(
        self,
        model: str = "sam-vit-large",
        device: str | None = None,
        cache_dir: str | None = None,
        gdino_config_path: str | None = None,
        gdino_checkpoint_path: str | None = None,
    ) -> None:
        if model in self._REMOVED:
            raise ValueError(f"Model '{model}' is not available: {self._REMOVED[model]}")
        self.model_name = model
        self.model_id = self.HF_MODELS.get(model, model)
        self.device = device or self._auto_device()
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._mask_generator = None
        self._chip_errors: list[str] = []
        # GroundingDINO (used by grounded_label) -- lazily loaded and cached.
        self._gdino_config_path = gdino_config_path
        self._gdino_checkpoint_path = gdino_checkpoint_path
        self._gdino_model = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoModelForMaskGeneration, AutoProcessor
        except ImportError as exc:
            raise ImportError(f"transformers + torch required: pip install transformers torch ({exc})") from exc
        logger.info("Loading SAM: %s -> %s", self.model_name, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = AutoModelForMaskGeneration.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model.to(self.device).eval()
        logger.info("SAM loaded")

    def _load_mask_generator(self):
        """Build (and cache) a "segment everything" pipeline reusing the loaded weights.

        ``transformers`` has no ``SamAutomaticMaskGenerator`` (that is Meta's separate
        ``segment_anything`` package); the equivalent is the ``mask-generation`` pipeline.
        """
        if self._mask_generator is not None:
            return self._mask_generator
        self._load()
        from transformers import pipeline
        self._mask_generator = pipeline(
            task="mask-generation",
            model=self._model,
            image_processor=self._processor.image_processor,
            device=self.device,
        )
        return self._mask_generator

    # ------------------------------------------------------------------ auto labelling

    @staticmethod
    def _read_rgb(src: Any, rgb_bands: tuple[int, int, int] | None):
        """Read three display bands as uint8, ignoring nodata when stretching."""
        import numpy as np

        if rgb_bands is None:
            rgb_bands = tuple(range(1, min(src.count, 3) + 1))  # type: ignore[assignment]
        if any(b < 1 or b > src.count for b in rgb_bands):
            raise ValueError(f"rgb_bands {rgb_bands} out of range for a {src.count}-band raster")
        data = src.read(list(rgb_bands), masked=True).astype("float32")
        out = np.zeros(data.shape, dtype="float32")
        for b in range(data.shape[0]):
            valid = data[b].compressed()
            if valid.size == 0:
                continue
            p2, p98 = np.percentile(valid, (2, 98))
            out[b] = np.clip((data[b].filled(p2) - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        if out.shape[0] == 1:
            out = np.repeat(out, 3, axis=0)
        elif out.shape[0] == 2:
            out = np.stack([out[0], out[1], out[0]], axis=0)
        return out[:3].transpose(1, 2, 0).astype(np.uint8)

    def auto_label(
        self,
        image_path: str | Path,
        output_path: str | Path = "./labels/sam_auto.tif",
        output_vector: str | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_area_m2: float = 10.0,
        max_area_m2: float | None = None,
        chip_size: int = 1024,
        overlap: int = 128,
        merge_overlapping: bool = True,
        rgb_bands: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Generate automatic segmentation masks from a GeoTIFF.

        Args:
            image_path: Input GeoTIFF.  SAM takes three display bands.
            output_path: Output instance-label GeoTIFF (0 = none, 1..N = instance id).
            output_vector: Optional GeoJSON of mask polygons.
            points_per_side: Grid density of point prompts per tile.
            pred_iou_thresh: Minimum predicted mask quality (IoU).
            stability_score_thresh: Minimum mask stability under a threshold shift.
            min_area_m2: Minimum mask ground area in square metres.
            max_area_m2: Maximum mask area (filters huge background).
            chip_size: Processing tile size in pixels.
            overlap: Overlap between tiles in pixels (must be smaller than ``chip_size``).
            merge_overlapping: Merge the same object detected in two overlapping tiles.
            rgb_bands: 1-based band numbers to use as (R, G, B).  Defaults to bands 1-3, which is
                only correct if the file is stored R, G, B -- Sentinel-2 files commonly are
                B2, B3, B4 (blue, green, red) and need ``rgb_bands=(3, 2, 1)``.

        Returns:
            Dict with n_masks, output_path, quality stats.  ``success`` is False when no mask
            survived or when tiles failed to process (``n_chip_errors``).
        """
        if chip_size < 64:
            raise ValueError(f"chip_size must be >= 64, got {chip_size}")
        if not 0 <= overlap < chip_size:
            raise ValueError(f"overlap must satisfy 0 <= overlap < chip_size, got {overlap} / {chip_size}")
        if points_per_side < 1:
            raise ValueError(f"points_per_side must be >= 1, got {points_per_side}")

        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import rasterio
        except ImportError as exc:
            raise ImportError(f"rasterio required: {exc}") from exc

        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            H, W = src.height, src.width
            rgb = self._read_rgb(src, rgb_bands)

        self._load()
        self._chip_errors = []
        t_start = time.time()

        all_masks: list[dict] = []
        n_tiles = 0
        for row in _tile_origins(H, chip_size, overlap):
            for col in _tile_origins(W, chip_size, overlap):
                chip = rgb[row:row + chip_size, col:col + chip_size]
                n_tiles += 1
                all_masks.extend(self._process_chip(
                    chip, points_per_side, pred_iou_thresh, stability_score_thresh, row=row, col=col))

        if merge_overlapping and overlap > 0:
            all_masks = self._merge_overlapping_masks(all_masks)

        pixel_area_m2 = _pixel_area_m2(transform, crs, H)
        filtered: list[dict] = []
        for m in all_masks:
            area_m2 = float(m["mask"].sum()) * pixel_area_m2
            if area_m2 < min_area_m2 or (max_area_m2 and area_m2 > max_area_m2):
                continue
            m["area_m2"] = area_m2
            filtered.append(m)
        logger.info("SAM: %d masks (of %d) passed filters", len(filtered), len(all_masks))

        # Paint large masks first so a small object inside a large one keeps its own id.
        label = np.zeros((H, W), dtype=np.uint32)
        for i, m in sorted(enumerate(filtered, start=1), key=lambda t: -t[1]["mask"].sum()):
            x1, y1, x2, y2 = m["bbox_full"]
            m["instance_id"] = i
            label[y1:y2, x1:x2][m["mask"]] = i

        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint32", compress="lzw", nodata=None)
        for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
            out_profile.pop(_k, None)
        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(label[np.newaxis])
            dst.update_tags(source="SAM_AutoLabel", model=self.model_name, n_masks=str(len(filtered)))

        vector_error = None
        if output_vector:
            try:
                self._masks_to_vector(filtered, transform, crs, output_vector)
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Mask vectorisation failed: %s", vector_error)

        n_failed = len(self._chip_errors)
        if n_failed:
            error: str | None = (
                f"{n_failed} of {n_tiles} tile(s) failed and are missing from the labels; "
                f"first error: {self._chip_errors[0]}")
        elif not filtered:
            error = (f"SAM found {len(all_masks)} raw mask(s) but 0 survived quality/area "
                     f"filtering -- the output raster contains no real labels.")
        else:
            error = None
        n = max(len(filtered), 1)
        return {
            "success": bool(filtered) and n_failed < n_tiles,
            "n_masks": len(filtered),
            "n_raw_masks": len(all_masks),
            "n_tiles": n_tiles,
            "n_chip_errors": n_failed,
            "output_path": str(output_path),
            "output_vector": None if vector_error else output_vector,
            "vector_error": vector_error,
            "duration_seconds": round(time.time() - t_start, 1),
            "model": self.model_name,
            "error": error,
            "quality_stats": {
                "mean_pred_iou": float(sum(m.get("predicted_iou", 0) for m in filtered) / n),
                # The pipeline applies the stability threshold internally and does not return
                # the per-mask stability value, so none is reported rather than a fake one.
                "stability_score_thresh": stability_score_thresh,
            },
        }

    @staticmethod
    def _make_entry(seg: Any, row: int, col: int, predicted_iou: float) -> dict | None:
        """Crop a tile-local boolean mask to its bounding box and place it in scene coordinates."""
        import numpy as np

        ys, xs = np.nonzero(seg)
        if ys.size == 0:
            return None
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        return {
            "mask": seg[y0:y1, x0:x1].copy(),
            "bbox_full": (col + x0, row + y0, col + x1, row + y1),
            "predicted_iou": predicted_iou,
        }

    def _merge_overlapping_masks(self, masks: list[dict], iou_threshold: float = 0.3) -> list[dict]:
        """Merge masks that are almost certainly the same object, found once per overlapping tile.

        Each mask is ``{"mask": bool crop, "bbox_full": (x1, y1, x2, y2), ...}``.  A cheap
        bounding-box test first, then the real IoU on the pixels in the two boxes' union.
        Merging is the union of the two masks, not an arbitrary pick of one.
        """
        if len(masks) < 2:
            return masks

        import numpy as np

        def _bbox_overlap(a, b):
            ax1, ay1, ax2, ay2 = a["bbox_full"]
            bx1, by1, bx2, by2 = b["bbox_full"]
            return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

        def _canvas(a, b):
            x1 = min(a["bbox_full"][0], b["bbox_full"][0])
            y1 = min(a["bbox_full"][1], b["bbox_full"][1])
            x2 = max(a["bbox_full"][2], b["bbox_full"][2])
            y2 = max(a["bbox_full"][3], b["bbox_full"][3])
            out = []
            for m in (a, b):
                c = np.zeros((y2 - y1, x2 - x1), dtype=bool)
                mx1, my1, mx2, my2 = m["bbox_full"]
                c[my1 - y1:my2 - y1, mx1 - x1:mx2 - x1] = m["mask"]
                out.append(c)
            return out[0], out[1], (x1, y1, x2, y2)

        merged_flags = [False] * len(masks)
        result = []
        for i in range(len(masks)):
            if merged_flags[i]:
                continue
            current = masks[i]
            for j in range(i + 1, len(masks)):
                if merged_flags[j]:
                    continue
                candidate = masks[j]
                if not _bbox_overlap(current, candidate):
                    continue
                ca, cb, box = _canvas(current, candidate)
                inter = np.logical_and(ca, cb).sum()
                union = np.logical_or(ca, cb).sum()
                if (inter / union if union > 0 else 0.0) >= iou_threshold:
                    current = dict(current)
                    current["mask"] = np.logical_or(ca, cb)
                    current["bbox_full"] = box
                    current["predicted_iou"] = max(
                        current.get("predicted_iou", 0), candidate.get("predicted_iou", 0))
                    merged_flags[j] = True
            result.append(current)

        if len(masks) > len(result):
            logger.info("SAM: merged %d duplicate mask(s) from tile overlap regions",
                        len(masks) - len(result))
        return result

    def _process_chip(
        self, chip: Any, points_per_side: int,
        pred_iou_thresh: float, stability_thresh: float,
        row: int = 0, col: int = 0,
    ) -> list[dict]:
        """Run SAM "segment everything" on one tile and return masks in scene coordinates.

        ``points_per_side``, ``pred_iou_thresh`` and ``stability_thresh`` are handed to the
        pipeline, which applies them (the previous version dropped all three, then re-filtered
        the *IoU* score against the *stability* threshold, which discarded most good masks).
        A failure is recorded in ``self._chip_errors`` and logged -- never silently turned into
        "no masks here".
        """
        try:
            from PIL import Image
            generator = self._load_mask_generator()
            outputs = generator(
                Image.fromarray(chip), points_per_batch=64, points_per_crop=points_per_side,
                pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_thresh,
            )
            entries = []
            for seg, score in zip(outputs["masks"], outputs["scores"]):
                entry = self._make_entry(_to_numpy_bool(seg), row, col, float(score))
                if entry is not None:
                    entries.append(entry)
            return entries
        except Exception as exc:
            msg = f"tile at row={row}, col={col}: {type(exc).__name__}: {exc}"
            logger.warning("SAM tile failed -- %s", msg)
            self._chip_errors.append(msg)
            return []

    # --------------------------------------------------------------- grounded labelling

    def _load_gdino(self) -> Any:
        """Load (and cache) the GroundingDINO model used by grounded_label().

        Uses the Swin-T OGC config shipped inside the ``groundingdino`` package and the
        matching checkpoint from the Hugging Face Hub, unless overridden by
        ``gdino_config_path`` / ``gdino_checkpoint_path``.
        """
        if self._gdino_model is not None:
            return self._gdino_model

        from groundingdino.util.inference import load_model as load_gdino

        config_path = self._gdino_config_path
        if config_path is None:
            import groundingdino
            config_path = str(
                Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py")

        checkpoint_path = self._gdino_checkpoint_path
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id="ShilongLiu/GroundingDINO",
                filename="groundingdino_swint_ogc.pth",
                cache_dir=self.cache_dir,
            )

        logger.info("Loading GroundingDINO: config=%s checkpoint=%s", config_path, checkpoint_path)
        self._gdino_model = load_gdino(config_path, checkpoint_path)
        return self._gdino_model

    @staticmethod
    def _gdino_image_tensor(rgb: Any) -> Any:
        """Preprocess an RGB array the way ``groundingdino.util.inference.load_image`` does.

        ``predict`` expects this normalised tensor, not a PIL image (passing the PIL image, as
        the previous version did, fails inside the model).
        """
        import groundingdino.datasets.transforms as GT
        from PIL import Image

        transform = GT.Compose([
            GT.RandomResize([800], max_size=1333),
            GT.ToTensor(),
            GT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor, _ = transform(Image.fromarray(rgb), None)
        return tensor

    @staticmethod
    def _gdino_boxes_to_pixels(boxes: Any, width: int, height: int) -> Any:
        """GroundingDINO returns normalised ``cx, cy, w, h``; SAM wants pixel ``x1, y1, x2, y2``."""
        import numpy as np

        b = np.asarray(boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else boxes,
                       dtype="float64").reshape(-1, 4)
        cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    def _segment_box(self, rgb: Any, box_xyxy: Any) -> Any | None:
        """Best SAM mask (H, W bool) for one pixel-space box, or ``None``."""
        import numpy as np
        import torch
        from PIL import Image

        self._load()
        inputs = self._processor(
            images=Image.fromarray(rgb), input_boxes=[[[float(v) for v in box_xyxy]]],
            return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**{k: v for k, v in inputs.items()
                                     if k in ("pixel_values", "input_boxes")}, multimask_output=True)
        if "reshaped_input_sizes" in inputs:  # SAM v1
            masks = self._processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())
        else:  # SAM2
            masks = self._processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
        cand = masks[0]                      # (point_batch, n_masks, H, W)
        cand = cand.reshape(-1, *cand.shape[-2:])
        iou = outputs.iou_scores.reshape(-1).cpu().numpy()
        if cand.shape[0] == 0:
            return None
        best = int(np.argmax(iou[: cand.shape[0]]))
        return _to_numpy_bool(cand[best])

    def grounded_label(
        self,
        image_path: str | Path,
        prompts: list[str],
        output_path: str | Path = "./labels/grounded_sam.tif",
        output_vector: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        rgb_bands: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Generate labels using GroundedSAM -- text-prompt driven segmentation.

        Combines Grounding DINO (text -> box) and SAM (box -> mask).  Class ``k`` in the
        output raster is ``prompts[k-1]``.  Requires the ``groundingdino`` package.
        """
        try:
            from groundingdino.util.inference import predict as gdino_predict
        except ImportError:
            return {"success": False, "error": "pip install groundingdino-py for GroundedSAM"}

        try:
            gdino_model = self._load_gdino()
        except ImportError as exc:
            return {"success": False, "error": f"GroundingDINO not available: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Failed to load GroundingDINO model/checkpoint: {exc}"}

        logger.info("GroundedSAM: prompts=%s", prompts)
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import rasterio

            self._load()
            with rasterio.open(str(image_path)) as src:
                profile = src.profile.copy()
                transform, crs = src.transform, src.crs
                H, W = src.height, src.width
                rgb = self._read_rgb(src, rgb_bands)

            tensor = self._gdino_image_tensor(rgb)
            label = np.zeros((H, W), dtype=np.uint8)
            detections: list[tuple[Any, int]] = []
            for class_idx, prompt in enumerate(prompts, start=1):
                boxes, _logits, _phrases = gdino_predict(
                    model=gdino_model, image=tensor, caption=prompt,
                    box_threshold=box_threshold, text_threshold=text_threshold,
                    device=self.device)
                for box in self._gdino_boxes_to_pixels(boxes, W, H):
                    detections.append((box, class_idx))

            kept: list[dict] = []
            for box, class_idx in detections:
                mask = self._segment_box(rgb, box)
                if mask is None or not mask.any():
                    continue
                label[mask] = class_idx
                entry = self._make_entry(mask, 0, 0, 0.0)
                if entry is not None:
                    entry["class_id"] = class_idx
                    kept.append(entry)

            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw", nodata=None)
            for _k in ("blockxsize", "blockysize", "tiled"):   # illegal without TILED=YES
                out_profile.pop(_k, None)
            with rasterio.open(str(output_path), "w", **out_profile) as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(source="GroundedSAM", prompts=str(prompts))
            vector_error = None
            if output_vector:
                try:
                    self._masks_to_vector(kept, transform, crs, output_vector)
                except Exception as exc:
                    vector_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Mask vectorisation failed: %s", vector_error)

            return {
                "success": len(kept) > 0,
                "n_prompts": len(prompts), "n_boxes_detected": len(detections),
                "n_masks_applied": len(kept), "output_path": str(output_path),
                "output_vector": None if (vector_error or not output_vector) else output_vector,
                "vector_error": vector_error,
                "error": None if kept else (
                    f"0 masks applied for prompts={prompts} -- Grounding DINO found "
                    f"{len(detections)} box(es), none produced a usable SAM mask."),
            }
        except Exception as exc:
            logger.exception("GroundedSAM failed")
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    def _masks_to_vector(self, masks, transform, crs, output_path):
        """Write mask polygons as WGS84 GeoJSON."""
        import json

        import numpy as np
        import rasterio.features
        from rasterio.warp import transform_geom
        from rasterio.windows import Window
        from rasterio.windows import transform as window_transform

        # RFC 7946 GeoJSON is WGS84 only; writing projected coordinates with no CRS would be
        # silently wrong in any GIS, so reproject.
        reproject = crs is not None and crs.to_epsg() != 4326
        features = []
        for i, m in enumerate(masks):
            x1, y1, x2, y2 = m["bbox_full"]
            local = window_transform(Window(x1, y1, x2 - x1, y2 - y1), transform)
            for geom, val in rasterio.features.shapes(m["mask"].astype(np.uint8), transform=local):
                if val == 1:
                    if reproject:
                        geom = transform_geom(crs, "EPSG:4326", geom)
                    features.append({
                        "type": "Feature", "geometry": geom,
                        "properties": {
                            "mask_id": m.get("instance_id", i),
                            "class_id": m.get("class_id"),
                            "area_m2": m.get("area_m2", 0),
                            "pred_iou": m.get("predicted_iou", 0),
                        },
                    })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)


class SamGeoLabeler:
    """Real wrapper around segment-geospatial (samgeo) -- the
    established, peer-reviewed reference integration of Meta's SAM/
    SAM2/SAM3 family for remote sensing (Wu & Osco 2023, JOSS;
    https://samgeo.gishub.org), confirmed installable
    (pip install segment-geospatial) and its real API verified
    directly against the installed package.

    This is a genuinely different tool from AutoLabeler/GroundedSAMLabeler
    above (this file's own from-scratch SAM wrappers): samgeo provides
    real tile-based generation for arbitrarily large GeoTIFFs, real
    direct-to-vector (GeoJSON/GeoPackage/Shapefile) export, and access
    to the full real SAM/SAM2/SAM3 model family through one actively-
    maintained package, rather than reimplementing that integration.

    Example::

        labeler = SamGeoLabeler(model_type="vit_h")
        result = labeler.label("scene.tif", output_path="./labels/sam_mask.tif")
    """

    def __init__(self, model_type: str = "vit_h", device: str | None = None,
                 checkpoint_dir: str | None = None) -> None:
        self.model_type = model_type
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self._sam = None

    def _load(self):
        if self._sam is not None:
            return self._sam
        try:
            from samgeo import SamGeo
        except ImportError as exc:
            raise ImportError(
                "SamGeoLabeler requires segment-geospatial: "
                "pip install segment-geospatial"
            ) from exc
        self._sam = SamGeo(
            model_type=self.model_type, automatic=True,
            device=self.device, checkpoint_dir=self.checkpoint_dir,
        )
        return self._sam

    def label(
        self,
        image_path: str | Path,
        output_path: str | Path = "./labels/sam_mask.tif",
        output_vector: str | None = None,
        foreground: bool = True,
        min_size: int = 0,
        **generate_kwargs,
    ) -> dict[str, Any]:
        """Generate real automatic segmentation masks via samgeo.

        Args:
            image_path: Source GeoTIFF.
            output_path: Output label raster path.
            output_vector: Optional path to also export real vector
                (GeoJSON/GeoPackage/Shapefile, inferred from extension)
                features via samgeo's real tiff_to_vector().
            foreground: Keep only foreground masks (samgeo's real
                foreground=True default).
            min_size: Minimum real mask size in pixels.
            **generate_kwargs: Forwarded to samgeo's real generate()
                (e.g. batch=True for tiled processing of large scenes).

        Returns:
            Dict with success, output_path, output_vector, error.
        """
        t0 = time.time()
        try:
            sam = self._load()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sam.generate(
                str(image_path), str(output_path), foreground=foreground,
                min_size=min_size, **generate_kwargs,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "duration_seconds": round(time.time() - t0, 1)}

        if not output_path.exists():
            return {
                "success": False,
                "error": f"samgeo.generate() completed without raising, but produced no "
                         f"real output file at {output_path} -- treat as a failed run.",
                "duration_seconds": round(time.time() - t0, 1),
            }

        vector_path, vector_error = None, None
        if output_vector:
            try:
                sam.tiff_to_vector(str(output_path), output_vector)
                vector_path = output_vector
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
                logger.warning("samgeo vector export failed: %s", exc)

        return {
            "success": True,
            "output_path": str(output_path),
            "output_vector": vector_path,
            "vector_error": vector_error,
            "model_type": self.model_type,
            "duration_seconds": round(time.time() - t0, 1),
        }
