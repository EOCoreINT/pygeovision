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


class SAMAutoLabeler:
    """Automated label generation using Segment Anything Model (SAM/SAM2).

    Zero-shot segmentation — generates high-quality binary or multi-class masks
    from satellite imagery without any training data.

    Supports:
        - Automatic mask generation (grid-based point prompts)
        - GroundedSAM (text-prompt driven)
        - SAM2 (video SAM for time-series)
        - Post-filtering by area, stability, IoU quality

    Example::

        labeler = SAMAutoLabeler(model="sam-vit-huge")
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
        "sam2-hiera-l":   "facebook/sam2-hiera-large",
        "sam2-hiera-b":   "facebook/sam2-hiera-base-plus",
        "geosam-vit-h":   "wangyi111/GeoSAM",
    }

    def __init__(
        self,
        model: str = "sam-vit-large",
        device: str | None = None,
        cache_dir: str | None = None,
        gdino_config_path: str | None = None,
        gdino_checkpoint_path: str | None = None,
    ) -> None:
        self.model_name = model
        self.model_id = self.HF_MODELS.get(model, model)
        self.device = device or self._auto_device()
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._mask_generator = None
        # GroundingDINO (used by grounded_label) — lazily loaded and cached.
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
        if self._model is not None:
            return
        try:
            from transformers import SamModel, SamProcessor
            logger.info("Loading SAM: %s → %s", self.model_name, self.device)
            self._processor = SamProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self._model = SamModel.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self._model.to(self.device)
            logger.info("SAM loaded")
        except ImportError as exc:
            raise ImportError(f"transformers + torch required: pip install transformers torch ({exc})")

    def _load_mask_generator(self):
        """Build (and cache) a "segment everything" pipeline reusing the
        already-loaded SamModel/SamProcessor weights.

        `transformers` does not expose `SamAutomaticMaskGenerator` — that
        class only exists in Meta's separate `segment_anything` package.
        The equivalent grid-of-points "segment everything" functionality is
        the mask-generation pipeline, built here from the already-loaded
        model/processor so weights aren't downloaded/loaded twice.
        """
        if self._mask_generator is not None:
            return self._mask_generator
        self._load()
        from transformers import pipeline
        self._mask_generator = pipeline(
            task="mask-generation",
            model=self._model,
            image_processor=self._processor.image_processor,
        )
        return self._mask_generator

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
    ) -> dict[str, Any]:
        """Generate automatic segmentation masks from a GeoTIFF.

        Uses grid-based SAM prompting with filtering by quality and area.

        Args:
            image_path: Input GeoTIFF (any number of bands)
            output_path: Output binary/instance label GeoTIFF
            output_vector: Optional GeoJSON of mask polygons
            points_per_side: Grid density for automatic prompting
            pred_iou_thresh: Minimum predicted IoU quality
            stability_score_thresh: Minimum mask stability score
            min_area_m2: Minimum mask area in square metres
            max_area_m2: Maximum mask area (filters huge background)
            chip_size: Processing tile size in pixels
            overlap: Overlap between tiles in pixels

        Returns:
            Dict with n_masks, output_path, quality stats
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import rasterio
            from PIL import Image
        except ImportError as exc:
            raise ImportError(f"rasterio + Pillow required: {exc}")

        # Load image
        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            H, W = src.height, src.width
            # Read RGB (first 3 bands) and normalise
            n_bands = min(src.count, 3)
            data = src.read(list(range(1, n_bands + 1))).astype(float)
            # Normalise each band to 0-255
            for b in range(data.shape[0]):
                p2, p98 = np.percentile(data[b], (2, 98))
                data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.stack([data[0], data[1], data[0]], axis=0)
            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

        # Load SAM model
        self._load()

        # Process in tiles
        all_masks = []
        t_start = time.time()
        stride = chip_size - overlap

        for row in range(0, H, stride):
            for col in range(0, W, stride):
                r2, c2 = min(row + chip_size, H), min(col + chip_size, W)
                chip = rgb[row:r2, col:c2]
                chip_masks = self._process_chip(
                    chip, points_per_side, pred_iou_thresh, stability_score_thresh
                )
                # Offset masks to full-image coordinates
                for m in chip_masks:
                    full_mask = np.zeros((H, W), dtype=bool)
                    ch, cw = m["segmentation"].shape
                    full_mask[row:row+ch, col:col+cw] = m["segmentation"]
                    m["full_mask"] = full_mask
                    m["bbox_full"] = (col, row, col+cw, row+ch)
                all_masks.extend(chip_masks)

        # Merge masks that were independently detected in the overlapping
        # region between adjacent tiles — without this, the same real
        # object gets counted as two separate mask instances near every
        # tile boundary, corrupting count/area statistics. Previously
        # accepted (merge_overlapping, defaulting to True) but never
        # referenced anywhere in this function.
        if merge_overlapping and overlap > 0:
            all_masks = self._merge_overlapping_masks(all_masks)

        # Filter by area
        pixel_area_m2 = abs(transform.a * transform.e)
        filtered_masks = []
        for m in all_masks:
            area_px = m["full_mask"].sum()
            area_m2 = area_px * pixel_area_m2
            if area_m2 < min_area_m2:
                continue
            if max_area_m2 and area_m2 > max_area_m2:
                continue
            m["area_m2"] = float(area_m2)
            filtered_masks.append(m)

        logger.info("SAM: %d masks (of %d) passed filters", len(filtered_masks), len(all_masks))

        # Build instance label raster
        label = np.zeros((H, W), dtype=np.uint32)
        for i, m in enumerate(filtered_masks, start=1):
            label[m["full_mask"]] = i

        # Save raster
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint32", compress="lzw")
        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(label[np.newaxis])
            dst.update_tags(
                source="SAM_AutoLabel",
                model=self.model_name,
                n_masks=str(len(filtered_masks)),
            )

        # Export vector
        if output_vector:
            self._masks_to_vector(filtered_masks, transform, crs, output_vector)

        return {
            "success": True,
            "n_masks": len(filtered_masks),
            "n_raw_masks": len(all_masks),
            "output_path": str(output_path),
            "output_vector": output_vector,
            "duration_seconds": round(time.time() - t_start, 1),
            "model": self.model_name,
            "quality_stats": {
                "mean_pred_iou": float(
                    sum(m.get("predicted_iou", 0) for m in filtered_masks) / max(len(filtered_masks), 1)
                ),
                "mean_stability": float(
                    sum(m.get("stability_score", 0) for m in filtered_masks) / max(len(filtered_masks), 1)
                ),
            },
        }

    def _merge_overlapping_masks(self, masks: list[dict], iou_threshold: float = 0.3) -> list[dict]:
        """Merge masks that are almost certainly the same real object,
        independently detected once per overlapping tile.

        A cheap bounding-box intersection check first (avoiding real
        IoU computation, which needs the full boolean arrays, for mask
        pairs nowhere near each other), then real IoU on the actual
        boolean masks for genuine candidates. Merging is a logical OR
        of the two masks (the union of both detections) rather than
        picking one arbitrarily — keeps whichever detection was more
        complete rather than truncating either.
        """
        if len(masks) < 2:
            return masks

        import numpy as np

        def _bbox_overlap(a, b):
            ax1, ay1, ax2, ay2 = a["bbox_full"]
            bx1, by1, bx2, by2 = b["bbox_full"]
            return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

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
                inter = np.logical_and(current["full_mask"], candidate["full_mask"]).sum()
                union = np.logical_or(current["full_mask"], candidate["full_mask"]).sum()
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    current = dict(current)
                    current["full_mask"] = np.logical_or(current["full_mask"], candidate["full_mask"])
                    current["predicted_iou"] = max(
                        current.get("predicted_iou", 0), candidate.get("predicted_iou", 0),
                    )
                    current["stability_score"] = max(
                        current.get("stability_score", 0), candidate.get("stability_score", 0),
                    )
                    merged_flags[j] = True
            result.append(current)

        n_merged = len(masks) - len(result)
        if n_merged > 0:
            logger.info("SAM: merged %d duplicate mask(s) from tile overlap regions", n_merged)
        return result

    def _process_chip(
        self, chip: Any, points_per_side: int,
        pred_iou_thresh: float, stability_thresh: float
    ) -> list[dict]:
        """Run SAM "segment everything" mode on one image chip via the
        transformers mask-generation pipeline.

        Note: unlike Meta's original `SamAutomaticMaskGenerator`, the
        transformers pipeline does not accept `pred_iou_thresh`/
        `stability_score_thresh`/`points_per_side` as constructor
        arguments — grid density is handled internally. We instead filter
        the returned per-mask IoU scores ourselves, and use that same score
        as a stability proxy since the pipeline doesn't expose the two
        separately.
        """
        try:
            from PIL import Image
            generator = self._load_mask_generator()
            img_pil = Image.fromarray(chip)
            outputs = generator(img_pil, points_per_batch=64)

            masks_out = []
            for seg, score in zip(outputs["masks"], outputs["scores"]):
                score = float(score)
                if score < pred_iou_thresh or score < stability_thresh:
                    continue
                masks_out.append({
                    "segmentation": seg,
                    "predicted_iou": score,
                    "stability_score": score,
                })
            return masks_out
        except Exception as exc:
            logger.debug("SAM chip failed: %s", exc)
            return []

    def _load_gdino(self) -> Any:
        """Load (and cache) the GroundingDINO model used by grounded_label().

        Uses the standard Swin-T OGC config shipped inside the
        `groundingdino` package by default, and auto-downloads the matching
        pretrained checkpoint from HuggingFace Hub if not already cached.
        Both can be overridden via `gdino_config_path`/`gdino_checkpoint_path`
        for custom deployments.
        """
        if self._gdino_model is not None:
            return self._gdino_model

        from groundingdino.util.inference import load_model as load_gdino

        config_path = self._gdino_config_path
        if config_path is None:
            import groundingdino
            config_path = str(
                Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
            )

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

    def grounded_label(
        self,
        image_path: str | Path,
        prompts: list[str],
        output_path: str | Path = "./labels/grounded_sam.tif",
        output_vector: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Generate labels using GroundedSAM — text-prompt driven segmentation.

        Combines Grounding DINO (text → bbox) + SAM (bbox → mask).

        Args:
            prompts: List of natural language prompts, e.g. ["building", "swimming pool"]
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
            return {"success": False,
                    "error": f"Failed to load GroundingDINO model/checkpoint: {exc}"}

        logger.info("GroundedSAM: prompts=%s", prompts)
        self._load()

        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import rasterio
            from PIL import Image
            with rasterio.open(str(image_path)) as src:
                profile = src.profile.copy()
                H, W = src.height, src.width
                data = src.read(list(range(1, min(src.count, 3) + 1))).astype(float)
                for b in range(data.shape[0]):
                    p2, p98 = np.percentile(data[b], (2, 98))
                    data[b] = np.clip((data[b]-p2)/(p98-p2+1e-8)*255, 0, 255)
                if data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

            label = np.zeros((H, W), dtype=np.uint8)
            all_boxes = []

            for class_idx, prompt in enumerate(prompts, start=1):
                # Grounding DINO detects boxes
                boxes, logits, phrases = gdino_predict(
                    model=gdino_model,
                    image=Image.fromarray(rgb),
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                all_boxes.extend([(b, class_idx, p) for b, p in zip(boxes, phrases)])

            # SAM segments each detected box
            if all_boxes:
                import torch
                processor = self._processor
                for box, class_idx, phrase in all_boxes:
                    inputs = processor(
                        images=Image.fromarray(rgb),
                        input_boxes=[[box.tolist()]],
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                    masks = processor.post_process_masks(
                        outputs.pred_masks, inputs["original_sizes"],
                        inputs["reshaped_input_sizes"]
                    )
                    if masks and masks[0].shape[0] > 0:
                        best = masks[0][0].cpu().numpy()
                        label[best > 0.5] = class_idx

            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw")
            with rasterio.open(str(output_path), "w", **out_profile) as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(source="GroundedSAM", prompts=str(prompts))

            return {"success": True, "n_prompts": len(prompts), "output_path": str(output_path)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _masks_to_vector(self, masks, transform, crs, output_path):
        """Convert mask list to GeoJSON polygons."""
        try:
            import json

            import numpy as np
            import rasterio.features

            features = []
            for i, m in enumerate(masks):
                shapes = list(rasterio.features.shapes(
                    m["full_mask"].astype(np.uint8), transform=transform
                ))
                for geom, val in shapes:
                    if val == 1:
                        features.append({
                            "type": "Feature",
                            "geometry": geom,
                            "properties": {
                                "mask_id": i,
                                "area_m2": m.get("area_m2", 0),
                                "pred_iou": m.get("predicted_iou", 0),
                                "stability": m.get("stability_score", 0),
                            },
                        })

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)
        except Exception as exc:
            logger.warning("Mask vectorisation failed: %s", exc)