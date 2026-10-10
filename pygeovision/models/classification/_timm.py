"""Shared timm construction for the classification builders."""
from __future__ import annotations

import logging
from typing import Any

from pygeovision.core.exceptions import ModelLoadError
from pygeovision.models import _adapt

logger = logging.getLogger(__name__)

# Errors that mean "the call was wrong", not "the weights could not be fetched".
_PROGRAMMING_ERRORS = (TypeError, ImportError, NotImplementedError)


def build_timm_classifier(
    timm_id: str,
    num_classes: int,
    in_channels: int,
    pretrained: bool,
    allow_random_init: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a timm classifier with ``num_classes`` outputs and ``in_channels`` bands.

    timm adapts the stem for ``in_channels != 3`` itself (it repeats and rescales the
    pretrained RGB filters).  If ``pretrained=True`` and the weights cannot be
    fetched, :class:`ModelLoadError` is raised -- the previous behaviour of logging a
    warning and returning random weights is available only with
    ``allow_random_init=True`` (and the model is flagged ``_pygeovision_is_random_init``).
    """
    import timm

    def make(with_weights: bool) -> Any:
        return timm.create_model(timm_id, pretrained=with_weights, num_classes=num_classes,
                                 in_chans=in_channels, **kwargs)

    if not pretrained:
        return make(False)
    try:
        return make(True)
    except _PROGRAMMING_ERRORS:
        raise
    except Exception as exc:
        if not allow_random_init:
            raise ModelLoadError(
                f"Could not load pretrained weights for timm model '{timm_id}': {exc}",
                details="Nothing was substituted. Pass pretrained=False for a randomly "
                        "initialised model, or allow_random_init=True to fall back to one.",
            ) from exc
        return _adapt.mark_random_init(make(False), f"pretrained load failed: {exc}")
