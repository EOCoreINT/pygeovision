"""U-Net for geospatial segmentation (segmentation-models-pytorch backed)."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_unet(
    backbone: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    decoder_channels: list[int] | None = None,
    **kwargs: Any,
) -> Any:
    """Build a U-Net with a configurable encoder.

    Args:
        backbone: Any smp encoder name, e.g. ``"resnet18"``, ``"resnet50"``,
            ``"resnet101"``, ``"efficientnet-b4"``.
        num_classes: Number of segmentation classes (output channels).
        in_channels: Input bands.  smp widens the first conv and, with pretrained
            weights, keeps the RGB filters (the extra bands are not pretrained).
        pretrained: Load ImageNet encoder weights (needs network on first use).
        decoder_channels: Decoder widths, default ``[256, 128, 64, 32, 16]``.

    Returns:
        ``smp.Unet``; ``model(x)`` returns ``(B, num_classes, H, W)`` logits.

    Without smp installed a plain 4-level U-Net is returned and ``backbone`` and
    ``pretrained`` are ignored -- a warning says so.

    Example::

        model = build_unet("resnet50", num_classes=7, in_channels=4)
    """
    decoder_channels = decoder_channels or [256, 128, 64, 32, 16]
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        logger.warning(
            "segmentation-models-pytorch is not installed: returning a plain U-Net; "
            "backbone=%r and pretrained=%r are IGNORED.  pip install "
            "segmentation-models-pytorch for encoder-based U-Nets.", backbone, pretrained,
        )
        return _build_simple_unet(in_channels, num_classes)
    return smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        decoder_channels=decoder_channels,
        **kwargs,
    )


def _build_simple_unet(in_ch: int, n_classes: int) -> Any:
    """A plain 4-level U-Net for environments without smp."""
    import torch
    import torch.nn as nn

    def _block(ci, co, p=1):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=p), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=p), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = _block(in_ch, 64)
            self.p1 = nn.MaxPool2d(2)
            self.e2 = _block(64, 128)
            self.p2 = nn.MaxPool2d(2)
            self.e3 = _block(128, 256)
            self.p3 = nn.MaxPool2d(2)
            self.e4 = _block(256, 512)
            self.p4 = nn.MaxPool2d(2)
            self.bn = _block(512, 1024)
            self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.d4 = _block(1024, 512)
            self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.d3 = _block(512, 256)
            self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.d2 = _block(256, 128)
            self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.d1 = _block(128, 64)
            self.out = nn.Conv2d(64, n_classes, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.p1(e1))
            e3 = self.e3(self.p2(e2))
            e4 = self.e4(self.p3(e3))
            b = self.bn(self.p4(e4))
            d4 = self.d4(torch.cat([self.u4(b), e4], 1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)

    return UNet()
