"""Detection model architectures (native registry).

Both builders delegate to :mod:`pygeovision.models.detection`, so the native and the
general registry build identical networks.  ``num_classes`` is the number of *object*
classes: a background slot is added, as for Faster R-CNN.  (An earlier version passed it
to torchvision unchanged, which meant "objects + background", only worked for
``in_channels == 3`` with ``pretrained=False``, and reported any other call as
"FCOS requires torchvision" because it swallowed every exception.)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_fcos(backbone: str = "resnet50", num_classes: int = 2, in_channels: int = 3,
               pretrained: bool = True, **kwargs: Any) -> Any:
    """Build an FCOS anchor-free object detector (ResNet-50 FPN)."""
    from pygeovision.models.detection.fcos import build_fcos as _build

    if backbone != "resnet50":
        raise ValueError(f"FCOS supports backbone='resnet50' only, got {backbone!r}")
    model = _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, **kwargs)
    logger.info("Built FCOS(num_classes=%d, in_channels=%d)", num_classes, in_channels)
    return model


def build_retinanet(backbone: str = "resnet50", num_classes: int = 2, in_channels: int = 3,
                    pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a RetinaNet object detector with focal loss (ResNet-50 FPN)."""
    from pygeovision.models.detection.retinanet import build_retinanet as _build

    if backbone != "resnet50":
        raise ValueError(f"RetinaNet supports backbone='resnet50' only, got {backbone!r}")
    model = _build(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, **kwargs)
    logger.info("Built RetinaNet(num_classes=%d, in_channels=%d)", num_classes, in_channels)
    return model
