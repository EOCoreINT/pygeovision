"""RetinaNet (ResNet-50 FPN) with focal loss -- torchvision backed."""
from __future__ import annotations

from typing import Any

from pygeovision.models.detection._tv import transform_kwargs, widen_detector_for_bands


def build_retinanet(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    **kwargs: Any,
) -> Any:
    """RetinaNet with the classification head resized to ``num_classes`` (+1 background).

    For ``in_channels != 3`` the backbone's first conv *and* the built-in normaliser are
    widened (see :func:`~pygeovision.models.detection._tv.widen_detector_for_bands`).
    """
    import torchvision
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    tv_kwargs = transform_kwargs(kwargs)
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        **tv_kwargs,
    )
    widen_detector_for_bands(model, in_channels, pretrained, image_mean, image_std)
    head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        model.backbone.out_channels, head.num_anchors, num_classes + 1,
    )
    return model
