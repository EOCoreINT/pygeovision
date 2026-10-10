"""DenseNet scene classifiers (timm)."""
from __future__ import annotations

from typing import Any

from pygeovision.models.classification._timm import build_timm_classifier


def build_densenet(variant: str = "121", num_classes: int = 10, in_channels: int = 4,
                   pretrained: bool = True, **kwargs: Any) -> Any:
    """DenseNet-121/161/169/201 classifier."""
    return build_timm_classifier(f"densenet{variant}", num_classes, in_channels, pretrained, **kwargs)
