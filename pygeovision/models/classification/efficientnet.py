"""EfficientNet scene classifiers (timm)."""
from __future__ import annotations

from typing import Any

from pygeovision.models.classification._timm import build_timm_classifier


def build_efficientnet(variant: str = "b4", num_classes: int = 10, in_channels: int = 4,
                       pretrained: bool = True, **kwargs: Any) -> Any:
    """EfficientNet-B0..B7 classifier, e.g. ``build_efficientnet("b4", 9, 4)``."""
    return build_timm_classifier(f"efficientnet_{variant}", num_classes, in_channels,
                                 pretrained, **kwargs)
