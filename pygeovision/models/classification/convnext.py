"""ConvNeXt scene classifiers (timm)."""
from __future__ import annotations

from typing import Any

from pygeovision.models.classification._timm import build_timm_classifier

_IDS = {"t": "convnext_tiny", "s": "convnext_small", "b": "convnext_base", "l": "convnext_large"}


def build_convnext(variant: str = "t", num_classes: int = 10, in_channels: int = 4,
                   pretrained: bool = True, **kwargs: Any) -> Any:
    """ConvNeXt tiny/small/base/large (``"t"|"s"|"b"|"l"``) classifier."""
    if variant not in _IDS:
        raise ValueError(f"variant must be one of {sorted(_IDS)}, got {variant!r}")
    return build_timm_classifier(_IDS[variant], num_classes, in_channels, pretrained, **kwargs)
