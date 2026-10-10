"""Swin Transformer scene classifiers (timm)."""
from __future__ import annotations

from typing import Any

from pygeovision.models.classification._timm import build_timm_classifier

_IDS = {
    "t": "swin_tiny_patch4_window7_224",
    "s": "swin_small_patch4_window7_224",
    "b": "swin_base_patch4_window7_224",
    "l": "swin_large_patch4_window7_224",
}


def build_swin(variant: str = "t", num_classes: int = 10, in_channels: int = 4,
               pretrained: bool = True, **kwargs: Any) -> Any:
    """Swin tiny/small/base/large classifier.

    Window attention is tied to the training resolution.  Pass ``img_size=<chip size>``
    (it must divide into the window grid) to use other sizes::

        model = build_swin("b", num_classes=20, in_channels=13, img_size=256)
    """
    if variant not in _IDS:
        raise ValueError(f"variant must be one of {sorted(_IDS)}, got {variant!r}")
    return build_timm_classifier(_IDS[variant], num_classes, in_channels, pretrained, **kwargs)
