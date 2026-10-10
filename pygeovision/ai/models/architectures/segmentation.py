"""
Segmentation model architectures for PyGeoVision.

Provides factory functions for building pixel-wise segmentation models:
U-Net, DeepLabV3+, SegFormer — all configurable for arbitrary numbers
of input bands and output classes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_unet(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    encoder_weights: str | None = "__unset__",
    activation: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build a U-Net segmentation model.

    Uses segmentation-models-pytorch (smp) as the backend.

    Args:
        encoder: Encoder backbone name (e.g. 'resnet50', 'efficientnet-b4').
        in_channels: Number of input channels (bands).
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained encoder weights.
            Real fix: this is the same convention every other build_*
            function in this package uses (build_resnet, build_vit,
            build_fcos, etc.) -- confirmed by direct testing that this
            parameter was previously silently ignored here (real weight
            downloads happened even when the caller explicitly requested
            pretrained=False, since this function only accepted
            encoder_weights, not pretrained).
        encoder_weights: Advanced override for a specific weight source
            string beyond True/False (e.g. a non-imagenet smp source).
            Takes precedence over `pretrained` only if explicitly set.
        activation: Output activation ('softmax2d', 'sigmoid', or None).
        **kwargs: Additional args forwarded to smp.Unet.

    Returns:
        smp.Unet model instance.

    Raises:
        ImportError: If segmentation-models-pytorch is not installed.

    Example:
        >>> model = build_unet(encoder="resnet50", in_channels=4, num_classes=10)
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "U-Net requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    if encoder_weights == "__unset__":
        encoder_weights = "imagenet" if pretrained else None

    # When in_channels != 3 and encoder_weights='imagenet', SMP patches
    # the first conv layer to accept arbitrary channels.
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
        **kwargs,
    )
    logger.info(
        "Built UNet(encoder=%s, in_channels=%d, num_classes=%d)",
        encoder, in_channels, num_classes,
    )
    return model


def build_deeplabv3plus(
    encoder: str = "resnet101",
    in_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    encoder_weights: str | None = "__unset__",
    encoder_output_stride: int = 16,
    **kwargs: Any,
) -> Any:
    """Build a DeepLabV3+ segmentation model.

    Args:
        encoder: Encoder backbone (e.g. 'resnet101', 'xception').
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained encoder weights.
            Real fix, same as build_unet: this was previously silently
            ignored (this function only accepted encoder_weights, not
            pretrained), confirmed by real weight downloads happening
            even with pretrained=False explicitly requested.
        encoder_weights: Advanced override beyond True/False. Takes
            precedence over `pretrained` only if explicitly set.
        encoder_output_stride: Output stride (16 or 8; 8 = higher resolution).
        **kwargs: Additional args for smp.DeepLabV3Plus.

    Returns:
        smp.DeepLabV3Plus model instance.

    Example:
        >>> model = build_deeplabv3plus(encoder="resnet101", num_classes=15)
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "DeepLabV3+ requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    if encoder_weights == "__unset__":
        encoder_weights = "imagenet" if pretrained else None

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        encoder_output_stride=encoder_output_stride,
        **kwargs,
    )
    logger.info(
        "Built DeepLabV3+(encoder=%s, in_channels=%d, num_classes=%d)",
        encoder, in_channels, num_classes,
    )
    return model


def build_segformer(
    model_size: str = "b2",
    in_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Build a SegFormer segmentation model.

    Args:
        model_size: SegFormer variant: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-21k pretrained weights.
        allow_random_init: If true, a failed pretrained download returns a flagged random
            model instead of raising.

    Returns:
        A module whose ``forward(x)`` returns ``(B, num_classes, H, W)`` logits at the input
        resolution.  The raw Hugging Face model emits 1/4 resolution and a ``ModelOutput``;
        the wrapper resizes and unwraps so the trainer, losses and tiled inference work
        unchanged.  ``pretrained=False`` builds offline.

    Example:
        >>> model = build_segformer(model_size="b5", num_classes=20)
    """
    from types import SimpleNamespace

    from pygeovision.models._hf import build_hf_model

    if model_size not in ("b0", "b1", "b2", "b3", "b4", "b5"):
        raise ValueError(
            f"model_size must be one of ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], got {model_size!r}"
        )
    hf_id = {"b5": "nvidia/segformer-b5-finetuned-ade-640-640"}.get(
        model_size, f"nvidia/segformer-{model_size}-finetuned-ade-512-512")
    spec = SimpleNamespace(name=f"segformer-{model_size}", family="segformer",
                           task="segmentation", hf_id=hf_id)
    allow_random_init = kwargs.pop("allow_random_init", False)
    if kwargs:
        raise TypeError(f"build_segformer got unexpected keyword arguments: {sorted(kwargs)}")
    model = build_hf_model(spec, num_classes, in_channels, pretrained,
                           allow_random_init=allow_random_init)
    logger.info("Built SegFormer-%s(in_channels=%d, num_classes=%d)",
                model_size.upper(), in_channels, num_classes)
    return model


def build_fpn(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> Any:
    """Build an FPN (Feature Pyramid Network) segmentation model.

    Args:
        encoder: Encoder backbone name.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_weights: Pretrained weights source.

    Returns:
        smp.FPN model instance.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "FPN requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    return smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )


def build_pan(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: str | None = "imagenet",
    **kwargs: Any,
) -> Any:
    """Build a PAN (Pyramid Attention Network) segmentation model.

    Args:
        encoder: Backbone encoder.
        in_channels: Input channels.
        num_classes: Output classes.
        encoder_weights: Pretrained source.

    Returns:
        smp.PAN model instance.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "PAN requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    return smp.PAN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
