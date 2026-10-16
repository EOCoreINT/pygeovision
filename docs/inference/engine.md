# The other inference engine

There is a second tiled engine, `pygeovision.ai.inference.TiledInference` (used by the `PyGeoVisionAIEngine` and the
model-based pipelines), with a different interface and slightly different behaviour. If you mix them you can get
different maps from the same checkpoint.

| | `pygeovision.inference.TiledInference` (this page) | `pygeovision.ai.inference.TiledInference` |
|---|---|---|
| Used by | `infer`, `ai infer`, `ai segment custom`, `client.segmentation.custom` | the AI engine and pipelines that take `--checkpoint` |
| Call | `.infer(path, out, normalise=…)` | `.run(path, out, num_classes=…)`; `.run_pair(...)` for two-date change models |
| Scaling | Percentile stretch of the scene, valid pixels, not clipped; switchable | Percentile stretch from a reduced-resolution read of the positive pixels, **clipped to 0–1**; not switchable (pass `preprocessing_fn` to add steps) |
| Reads | The whole scene | One window at a time from disk (the output accumulator is still full size) |
| No-data | Marked 255 | Not marked |
| Blend | `gaussian`, `linear`, `constant` | `gaussian`, `average` |

When a pipeline asks for `--checkpoint`, it uses the right-hand engine, so train on imagery scaled the same way: for each
band of each scene, the 2nd and 98th percentile of the pixels greater than zero become 0 and 1, and values are clipped
to that range.

```python
import numpy as np

def pipeline_stretch(band):                      # one band of one scene, as a 2-D array
    valid = band[band > 0]
    p2, p98 = np.percentile(valid, (2, 98)) if valid.size else (0.0, 1.0)
    return np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)
```

(The engine estimates the two percentiles from a reduced-resolution read of the scene, so tiny differences are
normal.) See [Pipelines](../pipelines/index.md) for which pipelines need a checkpoint.
