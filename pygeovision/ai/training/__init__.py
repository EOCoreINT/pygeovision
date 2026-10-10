"""PyGeoVision AI training package.

``GeoTrainer`` and ``TrainingConfig`` here are the one trainer. They are also
importable from ``pygeovision.training`` (same objects).
"""

try:
    from pygeovision.ai.training.callbacks import (
        Callback,
        EarlyStopping,
        LRSchedulerCallback,
        MLflowLogger,
        ModelCheckpoint,
    )
    from pygeovision.ai.training.losses import DiceFocalLoss, DiceLoss, FocalLoss, get_loss
    from pygeovision.ai.training.metrics import AverageMeter, BinaryMetrics, ConfusionMatrix
    from pygeovision.ai.training.trainer import CheckpointManager, GeoTrainer, TrainingConfig
    _TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    # torch is not installed: the trainer and the torch-based losses cannot be
    # imported. They are None rather than missing so ``import`` of this package works.
    _TORCH_AVAILABLE = False
    GeoTrainer = None  # type: ignore[assignment,misc]
    TrainingConfig = None  # type: ignore[assignment,misc]
    CheckpointManager = None  # type: ignore[assignment,misc]

    try:
        from pygeovision.ai.training.callbacks import (
            Callback,
            EarlyStopping,
            LRSchedulerCallback,
            MLflowLogger,
            ModelCheckpoint,
        )
        from pygeovision.ai.training.metrics import AverageMeter, BinaryMetrics, ConfusionMatrix
    except (ImportError, AttributeError):
        pass

from pygeovision.ai.training.distributed import (
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    setup_ddp,
    wrap_ddp,
)

__all__ = [
    "GeoTrainer", "TrainingConfig", "CheckpointManager",
    "get_loss", "DiceLoss", "FocalLoss", "DiceFocalLoss",
    "ConfusionMatrix", "BinaryMetrics", "AverageMeter",
    "Callback", "EarlyStopping", "ModelCheckpoint", "MLflowLogger", "LRSchedulerCallback",
    "setup_ddp", "cleanup_ddp", "wrap_ddp", "get_rank", "get_world_size", "is_main_process",
]
