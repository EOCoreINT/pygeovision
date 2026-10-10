"""ResNet scene classifiers (timm)."""
from __future__ import annotations

from typing import Any

from pygeovision.models.classification._timm import build_timm_classifier


def build_resnet(variant: str = "50", num_classes: int = 10, in_channels: int = 4,
                 pretrained: bool = True, **kwargs: Any) -> Any:
    """ResNet-18/34/50/101/152 classifier, e.g. ``build_resnet("50", 10, 13)``."""
    return build_timm_classifier(f"resnet{variant}", num_classes, in_channels, pretrained, **kwargs)
