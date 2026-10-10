"""Classification model architectures (native registry)."""
from __future__ import annotations

import logging
from typing import Any

from pygeovision.models import _adapt

logger = logging.getLogger(__name__)


def build_resnet(depth: int = 50, in_channels: int = 3, num_classes: int = 10,
                 pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a ResNet classifier (torchvision).

    ``pretrained=True`` loads ImageNet weights for *any* band count and then widens the
    first convolution (RGB filters kept, extra bands start from their mean).  Earlier
    versions dropped the pretrained weights altogether whenever ``in_channels != 3``,
    without saying so.
    """
    try:
        import torch.nn as nn
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError("ResNet requires torchvision. Install: pip install torchvision") from exc
    _map = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50,
            101: models.resnet101, 152: models.resnet152}
    if depth not in _map:
        raise ValueError(f"depth must be one of {list(_map.keys())}")
    model = _map[depth](weights="DEFAULT" if pretrained else None)
    if in_channels != 3:
        model.conv1 = _adapt.adapt_first_conv(model.conv1, in_channels, pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    logger.info("Built ResNet-%d(in_channels=%d, num_classes=%d)", depth, in_channels, num_classes)
    return model


def build_efficientnet(model_size: str = "b3", in_channels: int = 3, num_classes: int = 10,
                       pretrained: bool = True, **kwargs: Any) -> Any:
    """Build an EfficientNet classifier (timm).  Extra ``kwargs`` go to ``timm.create_model``."""
    try:
        from pygeovision.models.classification._timm import build_timm_classifier
        return build_timm_classifier(f"efficientnet_{model_size}", num_classes, in_channels,
                                     pretrained, **kwargs)
    except ImportError as exc:
        raise ImportError("EfficientNet requires timm. Install: pip install timm") from exc


def build_vit(model_size: str = "b16", in_channels: int = 3, num_classes: int = 10,
              pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a Vision Transformer (ViT) classifier (timm).

    Pass ``img_size=<chip size>`` for chips other than 224 px (timm resamples the position
    embedding when loading pretrained weights); it used to be silently dropped.
    """
    _name_map = {"b16": "vit_base_patch16_224", "b32": "vit_base_patch32_224",
                 "l16": "vit_large_patch16_224", "s16": "vit_small_patch16_224"}
    try:
        from pygeovision.models.classification._timm import build_timm_classifier
        return build_timm_classifier(_name_map.get(model_size, f"vit_{model_size}"), num_classes,
                                     in_channels, pretrained, **kwargs)
    except ImportError as exc:
        raise ImportError("ViT requires timm. Install: pip install timm") from exc
