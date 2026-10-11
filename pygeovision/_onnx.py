"""One place that calls ``torch.onnx.export`` the way every supported PyTorch accepts it.

PyTorch 2.6+ switched ``torch.onnx.export`` to a new exporter that needs the extra
package ``onnxscript``; without it the call dies with ``No module named 'onnxscript'``.
The classic exporter is still available with ``dynamo=False`` and needs nothing beyond
``onnx``. Older PyTorch does not know the ``dynamo`` argument at all, so on a
``TypeError`` the call is repeated without it.
"""
from __future__ import annotations

from typing import Any


def export_onnx(model: Any, dummy_input: Any, path: Any, **kwargs: Any) -> None:
    """``torch.onnx.export(model, dummy_input, path, **kwargs)`` with the classic exporter."""
    import torch

    try:
        torch.onnx.export(model, dummy_input, str(path), dynamo=False, **kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(model, dummy_input, str(path), **kwargs)
