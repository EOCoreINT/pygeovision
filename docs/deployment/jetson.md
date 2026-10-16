# NVIDIA Jetson and TensorRT

```python
from pygeovision.edge import JetsonDeployer

d = JetsonDeployer()
result = d.convert(model, "model.onnx", "model.trt",
                   precision="fp16",            # fp32 | fp16 | int8
                   input_shape=(1, 4, 512, 512),
                   workspace_gb=4)
```

`convert` does two things: it **exports a fixed-shape ONNX file** (TensorRT needs an optimisation profile for dynamic
dimensions, which this class does not create), then, if the `tensorrt` Python package is installed, builds the engine.

Without TensorRT (on any ordinary computer) you get the ONNX file and the command to finish the job on the device
(real output):

```text
{'success': False, 'stage': 'tensorrt_not_installed', 'onnx_path': 'm.onnx',
 'jetson_command': 'trtexec --onnx=m.onnx --saveEngine=m.trt --fp16 --memPoolSize=workspace:4096M',
 'note': 'Run the trtexec command on the Jetson device to complete conversion (TensorRT older than 8.4 uses --workspace=<MiB> instead of --memPoolSize)'}
```

Copy `m.onnx` to the Jetson, run that command there, and the engine file appears. `success: False` here means "the engine is not built
yet", not that the export failed. Other `stage` values: `onnx_export` (the export failed; `error` says why), `trt_parse`
(TensorRT could not read a layer; `errors` lists them), `trt_build` (the builder returned nothing; read TensorRT's log).

| Precision | Meaning | Caveat |
|---|---|---|
| `fp32` | Full precision | Slowest, safest |
| `fp16` | Half precision, if the GPU supports fast FP16 | Usually near-identical accuracy; the result reports `precision: fp32` and a note if the device lacks it |
| `int8` | 8-bit integers | **Built without calibration data**, so accuracy is not guaranteed and can collapse. Calibrate with representative imagery (with `trtexec`'s calibration options) and validate against the PyTorch model before use |

`JetsonDeployer().benchmark("model.trt", input_shape=(1, 4, 512, 512), n_runs=100)` times the engine (needs `tensorrt` and
`pycuda`; the output buffer is sized from the engine) and returns `mean_ms`, `median_ms`, `p95_ms`, `fps`, or `{"error": ...}`.

```{admonition} What has and has not been verified
:class: warning
The ONNX export, the fixed shape, the `trtexec` command and the no-TensorRT path were run. **The TensorRT build and the engine
benchmark were not**: no TensorRT or GPU was available. They follow the TensorRT 8.4-10 Python API (`set_memory_pool_limit`,
`build_serialized_network`, tensor-name queries with an older fallback) but treat the first run on a device as an experiment, and
check the engine's output against the PyTorch model. An engine is tied to the TensorRT version and the GPU that built it.
```
