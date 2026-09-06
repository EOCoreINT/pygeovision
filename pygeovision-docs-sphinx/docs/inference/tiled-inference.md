# Tiled Inference

Real inference on a satellite scene means tiling it into model-sized
chips, running the model on each, and blending overlapping predictions
back together. `pygeovision` has two real, genuinely separate
implementations of this — worth knowing about explicitly.

## Two real, separate engines

| Class | Reached via |
|---|---|
| `pygeovision.ai.inference.tiled_inference.TiledInference` | Every AI task pipeline's `_run_model` |
| `pygeovision.inference.tiled.TiledInference` | `pygeovision infer predict`/`infer batch` (CLI), and `pygeovision.inference.stream`/`edge.onnx_rt` (both delegate to it) |

These do not share code. A real band-count validation fix applied to
the first engine (see below) did not automatically apply to the
second — it needed its own, separate fix.

## A real, confirmed production bug: band-count mismatch

A real failure — "expected 3 bands, got 13" — traced back to a model
built for 3 input channels receiving a full, unfiltered multi-band
file. Both engines now run a real pre-flight check: a tiny dummy
forward pass with the actual selected band count, before committing to
the full tiled loop, so a mismatch surfaces immediately with a clear
message rather than failing deep inside the loop (or worse, silently
producing garbage if the model happens to accept the wrong channel
count without erroring).

```python
from pygeovision.inference.tiled import TiledInference

engine = TiledInference(model=model, num_classes=2, device="cpu")
result = engine.infer("scene.tif", "output.tif")
# {"success": False, "error": "Model does not accept 13 input band(s)
#  (selected from 13 real band(s) in scene.tif): ..."}
```

## CLI

```bash
pygeovision infer predict scene.tif --model unet_resnet50 --classes 2
pygeovision infer batch ./scenes/ ./predictions/ --model unet_resnet50
```

```{note}
Both commands previously defaulted `--model` to `"unet-r50"` — a
registry entry removed as fake during an earlier audit (it silently
returned a generic model regardless of the claimed backbone). Both now
route through `ModelHub.load()`, so they get the real fallback
mechanism described in [Model Registry](../core-features/model-registry.md)
rather than a single hardcoded name that could break the same way
again.
```
