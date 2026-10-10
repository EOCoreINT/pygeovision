"""GeoTrainer (Phase 4.1).

The implementation lives in :mod:`pygeovision.ai.training.trainer`. This module
re-exports it so ``pygeovision.training.GeoTrainer`` and
``pygeovision.ai.training.GeoTrainer`` are the same class. (They used to be two
diverging copies; fixes such as augmentation, detection mAP, backbone freezing and
checkpoint resume only reached one of them.)
"""
from pygeovision.ai.training.trainer import (  # noqa: F401
    CheckpointManager,
    EarlyStopping,
    GeoTrainer,
    TrainingConfig,
)

__all__ = ["GeoTrainer", "TrainingConfig", "EarlyStopping", "CheckpointManager"]
