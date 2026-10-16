#!/usr/bin/env python
"""Build every registered architecture (``pretrained=False``) and record what happens.

    PYTHONPATH=. python tools/probe_models.py            # writes docs/_data/model_probe.json

For each registry entry it records whether ``get_model`` builds it offline, the measured parameter count, the input it was
given and the shape that came out (or the error). The model pages in the documentation quote these numbers, so they are
measurements of this code, not copied from papers. Needs ``torch``; entries whose optional library is missing are recorded
with the error instead of being skipped.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_data" / "model_probe.json"
sys.path.insert(0, str(ROOT))


def _shape(x):
    import torch
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shape(i) for i in x[:3]]
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _shape(v) for k, v in list(x.items())[:2]}
    if hasattr(x, "logits"):
        return _shape(x.logits)
    return type(x).__name__


def probe(name: str, spec) -> dict:
    import torch
    from pygeovision.models import get_model
    rec: dict = {"task": spec.task, "family": spec.family}
    t0 = time.time()
    try:
        model = get_model(name, num_classes=3, in_channels=4, pretrained=False)
    except NotImplementedError as exc:
        return {**rec, "status": "listed_only", "message": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "status": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"}
    rec["build_seconds"] = round(time.time() - t0, 2)
    rec["class"] = type(model).__name__
    if isinstance(model, torch.nn.Module):
        rec["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        rec["status"] = "builds"
        try:
            model.eval()
            with torch.no_grad():
                if spec.task == "classification":
                    x = torch.rand(2, 4, 224, 224) if spec.family in ("vit", "swin", "dinov2") else torch.rand(2, 4, 64, 64)
                    rec["input"] = list(x.shape)
                    rec["output"] = _shape(model(x))
                elif spec.task == "segmentation":
                    x = torch.rand(2, 4, 64, 64)
                    rec["input"] = list(x.shape)
                    rec["output"] = _shape(model(x))
                elif spec.task == "change_detection":
                    a, b = torch.rand(2, 4, 64, 64), torch.rand(2, 4, 64, 64)
                    rec["input"] = [list(a.shape), list(b.shape)]
                    rec["output"] = _shape(model(a, b))
                elif spec.task == "detection" and spec.family == "detr":
                    x = torch.rand(1, 4, 128, 128)
                    rec["input"] = list(x.shape)
                    rec["output"] = _shape(model(x))
                elif spec.task == "detection":
                    x = [torch.rand(4, 128, 128)]
                    rec["input"] = "list of (4, 128, 128) tensors"
                    out = model(x)
                    rec["output"] = "list of dicts: " + ", ".join(sorted(out[0].keys())) if isinstance(out, list) and out and isinstance(out[0], dict) else _shape(out)
                elif spec.task == "3d":
                    model = get_model(name, num_classes=3, in_channels=3, pretrained=False).eval()
                    x = torch.rand(2, 3, 512)
                    rec["input"] = list(x.shape)
                    rec["output"] = _shape(model(x))
                elif spec.task == "super_resolution":
                    x = torch.rand(1, 4, 32, 32)
                    rec["input"] = list(x.shape)
                    rec["output"] = _shape(model(x))
        except Exception as exc:  # noqa: BLE001
            rec["forward_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    else:
        rec["status"] = "wrapper"
    return rec


def main() -> int:
    import logging
    logging.disable(logging.CRITICAL)
    from pygeovision.models.registry import _REGISTRY
    only = set(sys.argv[1:])
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, spec in _REGISTRY.items():
        if only and name not in only:
            continue
        try:
            data[name] = probe(name, spec)
        except Exception:  # noqa: BLE001
            data[name] = {"task": spec.task, "family": spec.family, "status": "error", "message": traceback.format_exc()[-300:]}
        print(name, data[name].get("status"), data[name].get("params_m", ""), data[name].get("output", data[name].get("message", ""))[:80] if isinstance(data[name].get("output", data[name].get("message", "")), str) else data[name].get("output"), flush=True)
    for gone in set(data) - set(_REGISTRY):
        del data[gone]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
