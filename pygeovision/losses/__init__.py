"""
PyGeoVision Geospatial Loss Functions (D1) — beyond generic cross-entropy.
All losses are implemented natively in pure PyTorch.
"""
from pygeovision.losses.class_balance import (
    ClassBalancedCrossEntropy,
    FocalCrossEntropy,
    LabelSmoothingCrossEntropy,
)
from pygeovision.losses.detection import (
    CIoULoss,
    DIoULoss,
    GIoULoss,
    SIoULoss,
)
from pygeovision.losses.segmentation import (
    BoundaryAwareLoss,
    ComboLoss,
    DiceLoss,
    FocalLoss,
    GeospatialMixedLoss,
    LovaszLoss,
    OhemCrossEntropy,
    TverskyLoss,
)

__all__ = [
    # Segmentation
    "DiceLoss", "FocalLoss", "TverskyLoss", "ComboLoss",
    "BoundaryAwareLoss", "LovaszLoss", "OhemCrossEntropy",
    "GeospatialMixedLoss",
    # Detection
    "CIoULoss", "DIoULoss", "GIoULoss", "SIoULoss",
    # Class balance
    "ClassBalancedCrossEntropy", "LabelSmoothingCrossEntropy", "FocalCrossEntropy",
]
