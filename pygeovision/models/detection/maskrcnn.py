"""Mask R-CNN (ResNet-50 FPN) for satellite instance segmentation -- torchvision backed."""
from __future__ import annotations

from typing import Any

from pygeovision.models.detection._tv import transform_kwargs, widen_detector_for_bands


def build_mask_rcnn(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    **kwargs: Any,
) -> Any:
    """Mask R-CNN with box and mask predictors resized to ``num_classes`` (+1 background)."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    tv_kwargs = transform_kwargs(kwargs)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        **tv_kwargs,
    )
    widen_detector_for_bands(model, in_channels, pretrained, image_mean, image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes + 1)
    return model
