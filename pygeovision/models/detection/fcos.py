"""FCOS (ResNet-50 FPN), an anchor-free detector -- torchvision backed."""
from __future__ import annotations

from typing import Any

from pygeovision.models.detection._tv import transform_kwargs, widen_detector_for_bands


def build_fcos(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    **kwargs: Any,
) -> Any:
    """FCOS with the classification head resized to ``num_classes`` (+1 background)."""
    import torchvision
    from torchvision.models.detection.fcos import FCOSClassificationHead

    tv_kwargs = transform_kwargs(kwargs)
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        **tv_kwargs,
    )
    widen_detector_for_bands(model, in_channels, pretrained, image_mean, image_std)
    model.head.classification_head = FCOSClassificationHead(
        model.backbone.out_channels, model.head.classification_head.num_anchors, num_classes + 1,
    )
    return model
