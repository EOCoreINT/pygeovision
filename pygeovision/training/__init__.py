"""
PyGeoVision Training Infrastructure (Phase 4).

Distributed training, HPO, experiment tracking, model optimisation, and serving.
"""
from pygeovision.training.experiment import ExperimentTracker
from pygeovision.training.hpo import ModelOptimizer, OptunaHPO
from pygeovision.training.metrics import (
    ChangeDetectionMetrics,
    DetectionMetrics,
    SegmentationMetrics,
)
from pygeovision.training.optimizer import build_optimizer, build_scheduler
from pygeovision.training.trainer import GeoTrainer, TrainingConfig

__all__ = [
    "GeoTrainer", "TrainingConfig",
    "build_optimizer", "build_scheduler",
    "SegmentationMetrics", "DetectionMetrics", "ChangeDetectionMetrics",
    "ExperimentTracker",
    "OptunaHPO", "ModelOptimizer",
]
