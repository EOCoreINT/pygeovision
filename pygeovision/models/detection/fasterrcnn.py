"""Faster R-CNN (ResNet-50 FPN) for satellite object detection -- torchvision backed."""
from __future__ import annotations

from typing import Any

from pygeovision.models.detection._tv import transform_kwargs, widen_detector_for_bands


def build_faster_rcnn(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    **kwargs: Any,
) -> Any:
    """Faster R-CNN with the box predictor resized to ``num_classes`` (+1 background).

    ``pretrained=True`` loads COCO detector weights and ImageNet backbone weights.
    For ``in_channels != 3`` the first conv and the built-in normaliser are widened
    (see :func:`widen_detector_for_bands`); the extra bands are *not* pretrained.
    Training/inference follow torchvision's API: ``model(images, targets)`` returns
    a loss dict in train mode, ``model(images)`` returns a list of dicts in eval mode.
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    tv_kwargs = transform_kwargs(kwargs)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        **tv_kwargs,
    )
    widen_detector_for_bands(model, in_channels, pretrained, image_mean, image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
