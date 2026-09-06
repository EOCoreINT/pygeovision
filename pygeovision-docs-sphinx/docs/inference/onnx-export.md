# ONNX Export

```python
from pygeovision.ai.training.export import ModelExporter

exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))
exporter.to_onnx("model.onnx")
```

Verified by loading the exported model back with `onnxruntime` and
comparing outputs numerically against the source PyTorch model
(`atol=1e-4`) — not just checking that export completes without
raising.

```{note}
Modern PyTorch's default ONNX exporter additionally requires
`onnxscript`, separately from the top-level `onnx` package. Without
it, export previously raised a raw, unwrapped `ModuleNotFoundError`
with no indication of what to install — found by hitting this
directly while verifying the export path. Now raises a clear message
naming the real fix: `pip install onnxscript`.
```

## Real inference with the exported model

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

engine = ONNXRuntimeInference("model.onnx")
result = engine.infer_geotiff("scene.tif", "prediction.tif")
```

`infer_geotiff` delegates to the same real `TiledInference` engine and
band-count pre-flight check described in
[Tiled Inference](tiled-inference.md) — it inherits that fix
automatically rather than needing a separate one.
