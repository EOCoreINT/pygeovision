"""FCN (fully convolutional network) for geospatial segmentation (torchvision backed)."""
from __future__ import annotations

from typing import Any


def build_fcn(
    backbone: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Build torchvision's FCN-ResNet; ``model(x)`` returns ``(B, num_classes, H, W)`` logits.

    ``pretrained=True`` loads ImageNet backbone weights (the COCO/VOC head is not
    reused: its class count differs).  torchvision's segmentation models do not
    normalise internally -- feed standardised input.
    """
    import torchvision

    from pygeovision.models import _adapt
    from pygeovision.models._wrappers import TorchvisionSegmentationAdapter

    builders = {
        "resnet50": torchvision.models.segmentation.fcn_resnet50,
        "resnet101": torchvision.models.segmentation.fcn_resnet101,
    }
    if backbone not in builders:
        raise ValueError(f"FCN backbone must be one of {sorted(builders)}, got {backbone!r}")
    model = builders[backbone](
        weights=None, weights_backbone="DEFAULT" if pretrained else None,
        num_classes=num_classes, aux_loss=False, **kwargs,
    )
    if in_channels != 3:
        model.backbone.conv1 = _adapt.adapt_first_conv(model.backbone.conv1, in_channels, pretrained)
    return TorchvisionSegmentationAdapter(model)
