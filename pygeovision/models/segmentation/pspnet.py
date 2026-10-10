"""PSPNet (pyramid pooling) for geospatial segmentation (segmentation-models-pytorch backed)."""
from __future__ import annotations

from typing import Any


def build_pspnet(
    backbone: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Build PSPNet; ``model(x)`` returns ``(B, num_classes, H, W)`` logits."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError("PSPNet needs segmentation-models-pytorch: "
                          "pip install segmentation-models-pytorch") from exc
    return smp.PSPNet(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
