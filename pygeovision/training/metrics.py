"""
Geospatial evaluation metrics (Phase 4.4).
IoU, F1, mAP, accuracy, confusion matrix for segmentation and detection.
"""
from __future__ import annotations

from typing import Any


class SegmentationMetrics:
    """Pixel-level segmentation metrics: mIoU, F1, accuracy, per-class IoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._confusion: Any | None = None
        self.n_failed_updates = 0
        self.reset()

    def reset(self) -> None:
        self.n_failed_updates = 0
        try:
            import torch
            self._confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        except ImportError:
            self._confusion = [[0] * self.num_classes for _ in range(self.num_classes)]

    def update(self, preds: Any, targets: Any) -> None:
        """Update confusion matrix with batch predictions and targets."""
        try:
            import torch
            preds = preds.reshape(-1)
            targets = targets.reshape(-1)
            mask = targets != self.ignore_index
            preds, targets = preds[mask], targets[mask]
            indices = self.num_classes * targets + preds
            counts = torch.bincount(indices.long(), minlength=self.num_classes ** 2)
            self._confusion += counts.reshape(self.num_classes, self.num_classes)
        except Exception as exc:
            import logging
            self.n_failed_updates += 1
            logging.getLogger(__name__).warning(
                "SegmentationMetrics.update() failed — this batch's "
                "predictions/targets were NOT counted, so mIoU/F1/accuracy "
                "will be computed from incomplete data. Common cause: preds "
                "contains class indices >= num_classes=%d, or preds/targets "
                "shape mismatch. Error: %s", self.num_classes, exc,
            )

    def compute(self) -> dict[str, float]:
        """Compute IoU, F1, accuracy from accumulated confusion matrix."""
        try:
            cm = self._confusion.float()
            tp = cm.diag()
            fp = cm.sum(0) - tp
            fn = cm.sum(1) - tp
            iou = tp / (tp + fp + fn + 1e-10)
            f1  = 2 * tp / (2 * tp + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            # Per-class numbers matter for rare classes: the mean over classes is
            # dominated by the (easy) background, so a model that finds no mining
            # at all still scores a high mean IoU.
            per_class = {}
            for i in range(self.num_classes):
                per_class[f"iou_class_{i}"] = float(iou[i])
                per_class[f"f1_class_{i}"] = float(f1[i])
                per_class[f"precision_class_{i}"] = float(precision[i])
                per_class[f"recall_class_{i}"] = float(recall[i])
            # Average over the classes that occur (in the reference or the prediction). A class
            # that is absent from both would count as IoU 0 and pull the mean down for no reason.
            present = (tp + fp + fn) > 0
            n_present = int(present.sum())
            return {
                "mean_iou":  float(iou[present].mean()) if n_present else 0.0,
                "mean_f1":   float(f1[present].mean()) if n_present else 0.0,
                "accuracy":  float(tp.sum() / (cm.sum() + 1e-10)),
                **per_class,
            }
        except Exception:
            return {"mean_iou": 0.0, "mean_f1": 0.0, "accuracy": 0.0}

    def confusion_matrix(self) -> Any:
        return self._confusion


class DetectionMetrics:
    """Object detection metrics: mAP@50, mAP@50-95 and mAP@75 (needs torchmetrics + pycocotools)."""

    def __init__(self, num_classes: int, iou_thresholds: list[float] | None = None) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self._predictions: list[dict] = []
        self._targets: list[dict] = []

    def reset(self) -> None:
        self._predictions.clear()
        self._targets.clear()

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        """
        predictions: list of dicts with 'boxes', 'scores', 'labels'
        targets:     list of dicts with 'boxes', 'labels'
        """
        self._predictions.extend(predictions)
        self._targets.extend(targets)

    def compute(self) -> dict[str, float]:
        """Compute mAP metrics."""
        if not self._predictions:
            return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            metric = MeanAveragePrecision(iou_thresholds=self.iou_thresholds)
            metric.update(self._predictions, self._targets)
            result = metric.compute()
            return {
                "mAP50":    float(result.get("map_50", 0.0)),
                "mAP50_95": float(result.get("map", 0.0)),
                "mAP75":    float(result.get("map_75", 0.0)),
            }
        except ImportError as exc:
            return {
                "mAP50": 0.0, "mAP50_95": 0.0,
                "note": (
                    f"{exc}. Install with: pip install torchmetrics[detection] "
                    f"(or pip install pycocotools)"
                ),
            }
        except Exception as exc:
            return {"mAP50": 0.0, "error": str(exc)}


class ChangeDetectionMetrics(SegmentationMetrics):
    """Binary change detection metrics: F1-change, precision, recall, IoU-change."""

    def __init__(self) -> None:
        super().__init__(num_classes=2)

    def compute(self) -> dict[str, float]:
        base = super().compute()
        return {
            "iou_change":       base.get("iou_class_1", 0.0),
            "f1_change":        base.get("mean_f1", 0.0),
            "mean_iou":         base.get("mean_iou", 0.0),
            "overall_accuracy": base.get("accuracy", 0.0),
        }
