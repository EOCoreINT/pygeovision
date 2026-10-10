"""Thin adapters that give third-party models the package's tensor-in/tensor-out contract.

``GeoTrainer``, ``TiledInference``, ONNX export and the loss functions all expect
``model(x)`` to return a plain ``(B, num_classes, H, W)`` tensor at the input
resolution.  Hugging Face segmentation models return a ``ModelOutput`` whose
logits are at a *fraction* of the input resolution (SegFormer: 1/4), and
torchvision's segmentation models return a dict.  Used unwrapped, a loss against
the full-resolution mask fails (or, worse, silently broadcasts), and tiled
inference stitches tiles at the wrong scale.  These wrappers fix that in one place.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(logits.shape[-2:]) != tuple(size):
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits


class HFSegmentationAdapter(nn.Module):
    """Hugging Face semantic-segmentation model -> ``(B, C, H, W)`` logits at input size."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)

    @property
    def hf_model(self) -> nn.Module:
        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return _resize_logits(logits, x.shape[-2:])


class HFClassificationAdapter(nn.Module):
    """Hugging Face image-classification model -> ``(B, num_classes)`` logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)

    @property
    def hf_model(self) -> nn.Module:
        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        return out.logits if hasattr(out, "logits") else out[0]


class TorchvisionSegmentationAdapter(nn.Module):
    """torchvision segmentation model (returns ``{"out": ...}``) -> logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: Any = self.model(x)
        logits = out["out"] if isinstance(out, dict) else out
        return _resize_logits(logits, x.shape[-2:])
