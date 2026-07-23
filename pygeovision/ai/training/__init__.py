"""PyGeoVision AI training package."""

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
    from pygeovision.ai.training.trainer import GeoTrainer, TrainingResult
    _TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    _TORCH_AVAILABLE = False
    GeoTrainer = None  # type: ignore[assignment,misc]
    TrainingResult = None  # type: ignore[assignment,misc]

    # Non-torch components — import unconditionally
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
    except Exception:
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
    "GeoTrainer", "TrainingResult",
    "get_loss", "DiceLoss", "FocalLoss", "DiceFocalLoss",
    "ConfusionMatrix", "BinaryMetrics", "AverageMeter",
    "Callback", "EarlyStopping", "ModelCheckpoint", "MLflowLogger", "LRSchedulerCallback",
    "setup_ddp", "cleanup_ddp", "wrap_ddp", "get_rank", "get_world_size", "is_main_process",
]
