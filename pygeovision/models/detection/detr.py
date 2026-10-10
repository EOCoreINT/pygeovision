"""DETR and RT-DETR object detectors (Hugging Face transformers).

These follow the Hugging Face detection API, not torchvision's: call
``model(pixel_values=x, labels=[{"class_labels": ..., "boxes": ...}])`` to get a loss and
``model(pixel_values=x)`` for ``logits`` / ``pred_boxes`` (boxes are normalised
``cx, cy, w, h``).  ``GeoTrainer``'s detection mode is torchvision-style
(Faster R-CNN / FCOS / Mask R-CNN), so train DETR with the Hugging Face loop or your own.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

_IDS = {
    "detr-r50": "facebook/detr-resnet-50",
    "detr-r101": "facebook/detr-resnet-101",
    "rt-detr-l": "PekingU/rtdetr_r50vd",
}


def build_detr(
    variant: str = "detr-r50",
    num_classes: int = 5,
    in_channels: int = 3,
    pretrained: bool = True,
    allow_random_init: bool = False,
    **kwargs: Any,
) -> Any:
    """Build DETR-R50 / DETR-R101 / RT-DETR.

    ``num_classes`` is the number of real object classes: DETR adds its own "no object"
    slot (an earlier version of this function added one on top, producing a phantom class).

    Raises:
        NotImplementedError: for ``rf-detr-*`` -- RF-DETR is distributed by Roboflow's
            ``rfdetr`` package, not as a transformers checkpoint, and has no module API here.
    """
    from pygeovision.models._hf import build_hf_model

    if variant.startswith("rf-detr"):
        raise NotImplementedError(
            "RF-DETR is distributed by Roboflow's `rfdetr` package (pip install rfdetr), "
            "not as a Hugging Face transformers checkpoint, and is not wrapped here."
        )
    if variant not in _IDS:
        raise ValueError(f"variant must be one of {sorted(_IDS)}, got {variant!r}")
    spec = SimpleNamespace(name=variant, family="detr", task="detection", hf_id=_IDS[variant])
    return build_hf_model(spec, num_classes, in_channels, pretrained,
                          allow_random_init=allow_random_init, **kwargs)
