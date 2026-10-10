"""Vision Transformer scene classifiers (Hugging Face transformers)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

_HF_IDS = {
    "b16": "google/vit-base-patch16-224",
    "l16": "google/vit-large-patch16-224",
}


def build_vit(
    variant: str = "b16",
    num_classes: int = 10,
    in_channels: int = 4,
    pretrained: bool = True,
    allow_random_init: bool = False,
    img_size: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build ViT-B/16 or ViT-L/16 as a classifier; ``model(x)`` returns ``(B, num_classes)`` logits.

    Bands other than three are handled by widening the patch embedding (extra bands are
    not pretrained).  ``img_size`` changes the expected chip size; with pretrained
    weights this re-initialises the position embedding (a warning is logged).
    """
    from pygeovision.models._hf import build_hf_model

    if variant not in _HF_IDS:
        raise ValueError(f"variant must be one of {sorted(_HF_IDS)}, got {variant!r}")
    spec = SimpleNamespace(name=f"vit-{variant}", family="vit", task="classification",
                           hf_id=_HF_IDS[variant])
    return build_hf_model(spec, num_classes, in_channels, pretrained,
                          allow_random_init=allow_random_init, img_size=img_size, **kwargs)
