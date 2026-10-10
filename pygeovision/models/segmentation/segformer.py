"""SegFormer (Mix Transformer encoder + all-MLP head) for geospatial segmentation."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# Encoder-only ImageNet checkpoints published by NVIDIA.  The decode head is
# initialised randomly and must be trained.
MIT_ENCODERS = {f"b{i}": f"nvidia/mit-b{i}" for i in range(6)}


def build_segformer(
    variant: str = "b2",
    num_classes: int = 2,
    in_channels: int = 4,
    pretrained: bool = True,
    allow_random_init: bool = False,
    **kwargs: Any,
) -> Any:
    """Build SegFormer-B0..B5.

    ``pretrained=True`` loads the ImageNet Mix-Transformer *encoder* (the decode
    head starts random); ``pretrained=False`` builds the architecture offline.
    Bands other than three are handled by widening the patch-embedding conv.

    Returns a module whose ``forward(x)`` gives ``(B, num_classes, H, W)`` logits at
    the input resolution (the raw model emits 1/4 resolution; the adapter resizes).

    Raises:
        ModelLoadError: pretrained weights could not be loaded (and
            ``allow_random_init`` is false).
    """
    from pygeovision.models._hf import build_hf_model

    if variant not in MIT_ENCODERS:
        raise ValueError(f"variant must be one of {sorted(MIT_ENCODERS)}, got {variant!r}")
    spec = SimpleNamespace(
        name=f"segformer-{variant}", family="segformer", task="segmentation",
        hf_id=MIT_ENCODERS[variant],
    )
    return build_hf_model(spec, num_classes, in_channels, pretrained,
                          allow_random_init=allow_random_init, **kwargs)
