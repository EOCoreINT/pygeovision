# Edge deployment: ONNX and Jetson

**Package:** `pygeovision.edge` · **Classes:** `ONNXRuntimeInference`, `JetsonDeployer` ·
**CLI:** `pygeovision edge export-onnx`, `pygeovision edge benchmark-onnx` · **Install:** `pip install "pygeovision[edge]"`
(`onnx`, `onnxruntime`, `onnxsim`; on a GPU machine install `onnxruntime-gpu` *instead of* `onnxruntime`)

## Why

PyTorch is large, slow to import, and tied to Python. The machine that has to run the model in the field often has
neither the disk space nor the patience: a ruggedised laptop, a drone companion computer, a server that runs thousands
of scenes and should not carry a deep-learning framework. **ONNX** is a neutral file format for a trained network;
**ONNX Runtime** executes it on CPU, NVIDIA GPU, Apple silicon and Jetson, without PyTorch. **TensorRT** goes further on
NVIDIA hardware by compiling the network for one specific GPU.

What you get: a portable file, a small runtime, often a faster forward pass. What you do **not** get: a more accurate
model, or any of the pre- and post-processing (that stays your job; see [Inference](../inference/index.md)).

## ONNX export and runtime

### Export

From the shell, for a model in the [registry](../models/registry.md) with weights you trained:

```console
$ pygeovision edge export-onnx unet-r50 --classes 3 --in-channels 4 --input-size 128 \
      --checkpoint runs/demo/checkpoints/best.pth -o model.onnx

  Exporting unet-r50 → ONNX (model.onnx)
  ✓ ONNX exported: model.onnx
```

| Option | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | | A registry name (`pygeovision models list`) |
| `--classes`, `--in-channels` | 2, 4 | Must equal what you trained with, or the checkpoint will not load |
| `--input-size` | 512 | The example input is `1 × in_channels × size × size`. This **fixes the file's input size** unless `--dynamic-size` |
| `--checkpoint` | none | Your trained weights, loaded **strictly**. Without it the registry's pretrained initialisation is exported (it downloads weights, needs a network, and is not your model) |
| `--dynamic-size` | off | Let the file accept any height and width |
| `-o`, `--output` | `./model.onnx` | |

From Python:

```python
from pygeovision.edge import ONNXRuntimeInference

engine = ONNXRuntimeInference.from_pytorch(
    model,                              # an nn.Module whose forward takes one tensor and returns one tensor
    "model.onnx",
    input_shape=(1, 4, 512, 512),       # (batch, bands, height, width) of the example
    opset=17,
    simplify=True,                      # runs onnxsim when installed; silently skipped when not
    dynamic_batch=True,                 # any batch size
    dynamic_spatial=False,              # True: any height and width as well
)
```

`from_pytorch` returns an engine already pointed at the new file. The `dynamic_spatial` option and `--dynamic-size`
exist because the default export is **fixed in height and width**:

| Export | Accepts | Use for |
|---|---|---|
| default | `batch × bands × H × W` with exactly the `H`, `W` you exported | Deployment at one chip size. Smallest risk, usually fastest |
| `dynamic_spatial=True` | any `H`, `W` | Tiling with different chip sizes, or whole small images |

On three registry networks (U-Net/ResNet-50, FCN-ResNet-50, SegFormer-B0, 4 bands, 3 classes) the dynamic export matched
PyTorch at 96, 128 and 160 pixels (largest difference 5·10⁻⁵ in the logits). Models with hard-wired reshapes or resizes
to a constant size can export wrongly with it, so always [check the export](#check-the-export) on a second size.

What the exporter needs from your model:

- `forward(x)` takes **one tensor** and returns **one tensor** (logits `(B, classes, H, W)`). Models that return a dict,
  a tuple or a Hugging Face output object must be wrapped first; the registry's adapters already do this.
- The graph is *traced*: Python `if` statements on tensor shapes are decided once, with the example input. The
  SegFormer export prints tracer warnings about this; they are expected when the decision does not depend on the data.
- Put the model in `eval()` mode (the exporter does), and on the CPU or GPU as you like (the example input follows the model).

Typical sizes (these networks, 4 input bands, 3 classes): U-Net/ResNet-50 130 MB, FCN-ResNet-50 132 MB, SegFormer-B0 15 MB.

### Run

```python
engine = ONNXRuntimeInference("model.onnx", device="cpu")     # cpu | cuda | tensorrt | coreml
logits = engine.infer(array)                                  # (C,H,W) or (B,C,H,W) → (B, classes, H, W) float32 logits
classes = logits.argmax(axis=1)
engine.model_info()                                           # input/output names, shapes, dtypes, active providers
```

| Constructor argument | Default | Notes |
|---|---|---|
| `device` | `"cpu"` | `"cuda"`, `"tensorrt"`, `"coreml"` ask for that execution provider and fall back to the CPU **with a warning** when it is not in your onnxruntime build (see below) |
| `optimization_level` | `"all"` | `none`, `basic`, `extended`, `all`: ONNX Runtime graph optimisations |
| `intra_op_threads` | 4 | Threads inside one operator. Set it to your number of physical cores; on a 2-core machine 4 oversubscribes |
| `inter_op_threads` | 1 | Parallel operators |

`infer` converts the input to `float32`, adds a batch axis to 3-D input and returns the **raw logits** of the first
output: apply `softmax` for probabilities and `argmax` for classes. It performs no scaling: feed the same normalised
values the model was trained on.

### Run on a whole GeoTIFF

```python
result = engine.infer_geotiff("scene.tif", "prediction.tif", chip_size=512, overlap=64,
                              blend_mode="gaussian", normalise=True)
```

This is the tiled engine of [Inference](../inference/index.md) with an ONNX forward pass, so tiling, Gaussian blending,
nodata handling and the class-map output are the same. A real run on a 256 × 256, 4-band scene with a dynamic SegFormer-B0 export:

```text
{'success': True, 'output_path': 'pred.tif', 'n_chips': 9, 'image_size': [256, 256], 'valid_fraction': 1.0,
 'duration_seconds': 0.6, 'chips_per_second': 14.4, 'blend_mode': 'gaussian', 'tta': False}
```

The output is a one-band `uint8` class map with the input's CRS and grid.

What the method does for you (all new behaviour, each caught a real failure):

- **`num_classes` is read from the model.** Leave it out (the default). If you pass a number that differs from the model's
  output channels you get `num_classes=5 but the model outputs 3 channels; pass num_classes=3 (or leave it out)`.
- **A fixed-size export adopts its own size.** If the file only accepts 128 × 128 and you ask for `chip_size=256`, the tiling
  uses 128 (and an overlap of at most a quarter of it) and logs a warning. Non-square fixed inputs are refused.
- **A test chip is run first**, so a wrong band count or a model that cannot run gives one clear error, not a failure
  deep inside the tile loop.

`normalise=True` stretches each band by percentiles for every scene, exactly as the PyTorch path does. If the model was trained on
other scaling (reflectance 0-1, standardised bands), pass `normalise=False` and make sure the data already has that scaling
([Matching the input to training](../inference/index.md#matching-the-input-to-training)).

### Check the export

Never ship an export you have not compared with the original:

```python
import numpy as np, torch

x = torch.rand(2, 4, 128, 128)
with torch.no_grad():
    ref = model.eval()(x).numpy()
out = ONNXRuntimeInference("model.onnx").infer(x.numpy())
print("max abs difference:", np.abs(ref - out).max(), "| same class:", (ref.argmax(1) == out.argmax(1)).mean())
```

Real results for the three registry networks above at 128 pixels:

| Network | Max abs difference of the logits | Pixels with the same class |
|---|---:|---:|
| U-Net / ResNet-50 | 6·10⁻⁵ | 100 % |
| FCN / ResNet-50 | 2·10⁻⁵ | 100 % |
| SegFormer-B0 | 9·10⁻⁸ | 100 % |

Differences around 10⁻⁵ are floating-point rounding. A difference of order 0.1 or more means the export is wrong; see
[what to do](#what-to-do-when).

### Benchmark

```console
$ pygeovision edge benchmark-onnx model.onnx --runs 30 [--device cuda]
  ONNX Benchmark (model.onnx):
  Mean: 24.0 ms | Median: 24.0 ms | P95: 40.9 ms | 38.3 FPS | providers: CPUExecutionProvider
```

In Python, `engine.benchmark(input_shape=None, n_runs=100)` returns `mean_ms`, `median_ms`, `p95_ms`, `fps` (from the mean), `n_runs`,
`input_shape`, `device` and `providers`. Five untimed runs warm the model up first. If `input_shape` is omitted it uses the
file's input shape, with a dynamic batch as 1 and any dynamic height or width as **512**.

Real numbers from a shared 2-core CPU (so slow, and noisy), 128 × 128 chips, 4 bands, batch 1:

| Network | Parameters | Median / mean | 95th percentile | Note |
|---|---:|---:|---:|---|
| SegFormer-B0 | 3.7 M | 24 / 26 ms | 41 ms | |
| U-Net / ResNet-50 | 32.5 M | 73-100 ms | 103-133 ms | two runs of the same file disagreed by 25 % |
| FCN / ResNet-50 | 33.0 M | 180 / 184 ms | 219 ms | |
| SegFormer-B0, **dynamic** export, default shape | 3.7 M | 411 ms | 482 ms | benchmarked at the 512 × 512 default, not 128 |

Use these as an illustration of the method, not as figures for your hardware:

- **Compare like with like.** The last row looks 16 times slower only because the benchmark used a larger input.
- **Look at the median and the 95th percentile**, not just the mean; a shared machine produces outliers.
- **The benchmark times the model only**: not reading the file, scaling the bands, tiling or writing the result.
- **Run on the target hardware** with its real thread count. A laptop CPU tells you nothing about a Jetson.

### Execution providers

| `device` | Provider requested | Needs |
|---|---|---|
| `cpu` | `CPUExecutionProvider` | `onnxruntime` |
| `cuda` | `CUDAExecutionProvider`, then CPU | `onnxruntime-gpu` and a matching CUDA/cuDNN install |
| `tensorrt` | `TensorrtExecutionProvider` (FP16 on), then CUDA, then CPU | `onnxruntime-gpu` built with TensorRT; the first run compiles the engine and is slow |
| `coreml` | `CoreMLExecutionProvider`, then CPU | `onnxruntime` for Apple silicon |

If the requested provider is not available the session still loads and runs on the CPU, and the log says so:

```text
WARNING device='cuda' was requested but CUDAExecutionProvider is not available in this onnxruntime build
(active: ['CPUExecutionProvider']); running on the CPU. Install onnxruntime-gpu ...
```

`engine.benchmark()["providers"]` and `engine.model_info()["providers"]` show what is actually used. The CUDA, TensorRT and CoreML
paths could not be run in PyGeoVision's own test environment (CPU only).

## NVIDIA Jetson and TensorRT

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

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `Got invalid dimensions for input: index: 2 Got: 32 Expected: 128` | The file was exported at a fixed size | Feed that size; or re-export with `--dynamic-size` / `dynamic_spatial=True` |
| `infer_geotiff` logs "the model was exported for 128x128 inputs: using chip_size=128" | You asked for another chip size | Expected; re-export with `dynamic_spatial=True` to use yours |
| `num_classes=… but the model outputs N channels` | Wrong `num_classes` | Use `N`, or omit it |
| `the model failed on a 512x512 test chip with 4 bands` | Wrong band count for this file, or a size it cannot take | Match the band count; read the message after the colon |
| `Got invalid dimensions for input: index: 1 Got: 3 Expected: 4` | Image has 3 bands, model expects 4 | Supply 4 bands in training order |
| Export fails with `No module named 'onnxscript'` | A new PyTorch exporter was used | PyGeoVision uses the classic exporter (`dynamo=False`); you are probably calling `torch.onnx.export` yourself |
| Export fails on a Hugging Face model | The model returns an output object | Use the registry model (it is wrapped) or wrap yours to return one tensor |
| `Cannot find file …` / `NO_SUCHFILE` | Wrong path | Check the path |
| Exported output differs by about 0.1 or more | Dynamic-shape trap, unsupported operator, or the model was left in train mode | Export with the default fixed size; call `model.eval()`; compare layer by layer |
| `device='cuda' was requested but CUDAExecutionProvider is not available` | CPU-only onnxruntime, or no GPU driver | `pip uninstall onnxruntime && pip install onnxruntime-gpu`, check CUDA |
| Latency is far worse than expected | Fewer cores than `intra_op_threads`; sharing the machine; benchmarking a bigger default shape | Set threads; pass `input_shape=`; use the median |
| `ValueError: n_runs must be at least 1` | `--runs 0` | Use 1 or more |
| `{'success': False, 'stage': 'tensorrt_not_installed', …}` | TensorRT is not installed here | Run the `jetson_command` on the Jetson |
| `trtexec` rejects `--workspace=…` | Newer TensorRT replaced it with `--memPoolSize` | Use `--memPoolSize=workspace:4096M` (the generated command already does; on TensorRT older than 8.4 use `--workspace=4096`) |

```{seealso}
[Cloud deployment](cloud.md) · [Exporting and shrinking a model](../training/advanced.md#exporting-and-shrinking-a-model) ·
[Inference](../inference/index.md) · [Benchmarking](../benchmark/index.md)
```
