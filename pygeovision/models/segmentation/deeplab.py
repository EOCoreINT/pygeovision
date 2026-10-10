"""DeepLabV3 / DeepLabV3+ for geospatial segmentation (segmentation-models-pytorch backed)."""
from __future__ import annotations

from typing import Any


def build_deeplab(
    backbone: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    plus: bool = True,
    **kwargs: Any,
) -> Any:
    """Build DeepLabV3+ (``plus=True``, default) or DeepLabV3 (``plus=False``).

    Both use atrous convolution for multi-scale context.  ``model(x)`` returns
    ``(B, num_classes, H, W)`` logits.  Requires segmentation-models-pytorch;
    unlike U-Net there is no plain fallback, so a missing install raises.

    Example::

        model = build_deeplab("resnet101", num_classes=11)
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError("DeepLab needs segmentation-models-pytorch: "
                          "pip install segmentation-models-pytorch") from exc
    cls = smp.DeepLabV3Plus if plus else smp.DeepLabV3
    return cls(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
