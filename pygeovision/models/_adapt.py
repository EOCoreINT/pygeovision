"""Input-channel adaptation shared by every model builder.

Satellite imagery rarely has three bands.  Pretrained backbones expect RGB, so
taking one to 4, 6, 10 or 13 bands needs two things, and the most common bug is
doing only the first:

1. The first convolution has to accept the new channel count
   (:func:`adapt_first_conv`).
2. Anything that normalises the input *before* the network has to know the new
   channel count too.  torchvision's detection models normalise inside the
   model with a 3-element mean/std, so a 4-band image crashes in ``forward``
   even though the backbone itself was widened (:func:`set_detection_normalisation`).

Hugging Face vision models add a third wrinkle: the patch-embedding layer
and a ``num_channels`` attribute that is checked on every forward pass
(:func:`adapt_hf_input_channels`).

None of this can give the new bands *pretrained meaning*.  ImageNet never saw
a SWIR band.  The extra channels start from the mean of the RGB filters, which
is a sensible starting point for fine-tuning, not a pretrained representation;
every function here logs that.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def adapt_first_conv(conv: Any, in_channels: int, pretrained_loaded: bool) -> Any:
    """Return a copy of ``conv`` that takes ``in_channels`` input channels.

    When ``pretrained_loaded`` is true the first three filters are copied, and
    any extra channels start from the mean of the pretrained filters (fewer than
    three channels take the first ``in_channels`` filters).  Otherwise the new
    convolution keeps PyTorch's default initialisation.  Stride, padding,
    dilation, groups, bias and padding mode are preserved.
    """
    import torch
    import torch.nn as nn

    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    if in_channels == conv.in_channels:
        return conv
    if conv.groups != 1:
        raise ValueError(
            f"cannot adapt a grouped convolution (groups={conv.groups}) to a new "
            f"channel count"
        )

    new_conv = nn.Conv2d(
        in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        bias=conv.bias is not None, padding_mode=conv.padding_mode,
    )
    new_conv = new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
    if pretrained_loaded:
        with torch.no_grad():
            k = min(3, in_channels, conv.in_channels)
            new_conv.weight[:, :k] = conv.weight[:, :k]
            if in_channels > k:
                mean_w = conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, k:] = mean_w.expand(-1, in_channels - k, -1, -1)
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        logger.warning(
            "First conv adapted from %d to %d input channels: pretrained filters kept "
            "for the first %d channel(s); the other channel(s) start from the mean "
            "pretrained filter.  They are NOT pretrained on those bands -- fine-tune.",
            conv.in_channels, in_channels, k,
        )
    return new_conv


def extend_stats(values: tuple[float, ...] | list[float], n: int) -> list[float]:
    """Extend per-channel statistics to ``n`` channels (extras use the mean)."""
    values = list(values)
    if n <= len(values):
        return values[:n]
    fill = sum(values) / len(values)
    return values + [fill] * (n - len(values))


def set_detection_normalisation(
    model: Any,
    in_channels: int,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> Any:
    """Make a torchvision detection model's built-in normaliser ``in_channels`` wide.

    torchvision's ``GeneralizedRCNNTransform`` subtracts a 3-element mean, so any
    other band count fails inside ``forward``.  Defaults are the ImageNet
    statistics for the first three channels and their mean for extra channels;
    pass ``image_mean`` / ``image_std`` (length ``in_channels``) to use the real
    statistics of your sensor instead.
    """
    transform = getattr(model, "transform", None)
    if transform is None or not hasattr(transform, "image_mean"):
        return model
    mean = list(image_mean) if image_mean is not None else extend_stats(IMAGENET_MEAN, in_channels)
    std = list(image_std) if image_std is not None else extend_stats(IMAGENET_STD, in_channels)
    if len(mean) != in_channels or len(std) != in_channels:
        raise ValueError(
            f"image_mean/image_std must have {in_channels} values each, "
            f"got {len(mean)} and {len(std)}"
        )
    transform.image_mean = mean
    transform.image_std = std
    return model


def adapt_hf_input_channels(model: Any, in_channels: int, pretrained_loaded: bool) -> Any:
    """Widen a Hugging Face vision model's stem to ``in_channels`` bands (in place).

    Finds the first 3-channel ``Conv2d`` (the patch-embedding / stem), replaces it
    through :func:`adapt_first_conv`, and updates every ``num_channels`` attribute
    that the model validates on each forward pass.
    """
    import torch.nn as nn

    if in_channels == 3:
        return model
    target_name, target = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            target_name, target = name, module
            break
    if target is None:
        raise ValueError(
            f"{type(model).__name__} has no 3-channel convolution to adapt to "
            f"{in_channels} input channels"
        )
    parent = model
    *path, leaf = target_name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, leaf, adapt_first_conv(target, in_channels, pretrained_loaded))

    for module in model.modules():
        if getattr(module, "num_channels", None) == 3:
            module.num_channels = in_channels
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "num_channels", None) is not None:
        cfg.num_channels = in_channels
    return model


def mark_random_init(model: Any, reason: str) -> Any:
    """Flag a model that is NOT carrying the pretrained weights that were requested."""
    try:
        model._pygeovision_is_random_init = True
        model._pygeovision_random_init_reason = reason
    except Exception:  # pragma: no cover - exotic frozen objects
        pass
    logger.warning("Returning a RANDOMLY INITIALISED model: %s", reason)
    return model
