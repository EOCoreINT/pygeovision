"""Shared pieces for the torchvision detectors (Faster R-CNN, Mask R-CNN, FCOS)."""
from __future__ import annotations

import logging
from typing import Any

from pygeovision.models import _adapt

logger = logging.getLogger(__name__)


def widen_detector_for_bands(
    model: Any,
    in_channels: int,
    pretrained: bool,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> Any:
    """Make a torchvision detector accept ``in_channels`` bands.

    Two things must change, and the second is easy to miss: the backbone's first
    convolution, *and* the model's built-in normaliser (``model.transform``),
    which is hard-wired to three channels.  Without the second, ``forward`` raises
    ``RuntimeError: The size of tensor a (4) must match the size of tensor b (3)``.
    """
    if in_channels != 3:
        body = model.backbone.body
        body.conv1 = _adapt.adapt_first_conv(body.conv1, in_channels, pretrained)
    _adapt.set_detection_normalisation(model, in_channels, image_mean, image_std)
    return model


def transform_kwargs(kwargs: dict) -> dict:
    """Pull torchvision ``GeneralizedRCNNTransform`` options out of ``kwargs``.

    torchvision resizes every image so its short side is ``min_size`` (800 by
    default) -- a 256 px satellite chip is upsampled ~3x.  Pass ``min_size`` /
    ``max_size`` (e.g. the chip size) to keep native resolution.
    """
    return {k: kwargs.pop(k) for k in ("min_size", "max_size") if k in kwargs}
