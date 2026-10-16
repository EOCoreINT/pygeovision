# Exporting and shrinking a model

**Why.** The `.pth` file needs PyTorch and your code to load. Exporting makes a file other tools can run (ONNX Runtime
on a server or an edge device, TensorRT, a browser), and compression makes it smaller or faster at some cost in
accuracy.

## ONNX

```python
import torch
from pygeovision.models import get_model
from pygeovision.training.hpo import ModelOptimizer

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("runs/demo/checkpoints/best.pth", weights_only=True)["model_state_dict"])

opt = ModelOptimizer(model, num_classes=2, in_channels=4)
path = opt.export_onnx("model.onnx", input_size=512)       # any batch size, exactly 512 x 512 pixels; needs `onnx`
out = opt.onnx_inference(some_array, path)                 # (4, H, W) or (1, 4, H, W) float32 array; needs onnxruntime
```

Expected: `model.onnx` is written (a few MB for a small net, over 100 MB for U-Net/ResNet-50), `onnx_inference` returns an
array shaped `(1, classes, H, W)` that matches the PyTorch output to about 1e-4. Pass `simplify=False` to skip graph
simplification (it runs only if `onnxsim` is installed). **The file accepts exactly the `input_size` you export with**
(any batch size, but one height and width): another size fails with `Got invalid dimensions for input`. Pass
`dynamic_spatial=True` to leave height and width free; it worked for U-Net/ResNet-50, FCN-ResNet-50 and SegFormer-B0
(outputs matched PyTorch at 96, 128 and 160 pixels), but check your own architecture on a second size, because models
with fixed reshapes or resizes can export wrongly. Deployment details are in [Edge deployment](../deployment/edge.md).

The same can be done from the shell for a registry model with weights you trained:

```bash
pygeovision edge export-onnx unet-r50 --checkpoint runs/demo/checkpoints/best.pth \
        --classes 2 --in-channels 4 --input-size 512 -o model.onnx      # add --dynamic-size for any height/width
```

`TrainingConfig(export_onnx=True)` also writes `model.onnx` after training, but of the **last epoch**, not the best.
Prefer exporting `best.pth` as above. Older PyTorch exporters needed an extra package (`onnxscript`); PyGeoVision
uses the classic exporter, so only `onnx` is required.

## TorchScript

`opt.export_torchscript("model.pt", method="trace", input_size=512)` writes a self-contained file that loads with
`torch.jit.load` and no source code. Tracing fixes the control flow for the example input; if your model branches on
input values use `method="script"`.

## INT8 quantisation

```python
q = ModelOptimizer(model, in_channels=4).quantize_int8(calibration_loader, "model_int8.pth")
```

The calibration loader should yield batches of **real, training-like** chips (`(images, labels)` pairs are fine). The
result runs on a **CPU** with `float32` input and output. Layers are not fused, so the speed-up and accuracy are
modest. Many segmentation networks use operations PyTorch cannot quantise (upsampling, concatenation, some
activations). In that case you get a `RuntimeError` saying so rather than a model that fails later: export to ONNX and
quantise there. **Always compare IoU on validation data before and after**; a quantised model that runs is not
necessarily one that is still accurate.

## Pruning

`opt.prune(amount=0.3)` sets the 30% smallest weights of every convolution and linear layer to zero (`method="global"`
prunes across the whole network instead). Batch-norm layers are left alone. The model keeps its ordinary weight names,
so it loads into the original architecture. Zeros alone do not shrink the file or speed up ordinary dense inference; use
pruning as a starting point for fine-tuning or a sparse runtime. Fine-tune afterwards to recover accuracy.

## Speed

`opt.benchmark_speed(input_size=512, n_runs=100, device="cuda")` returns `{"mean_ms", "std_ms", "fps", "p95_ms", ...}`
for one image (falls back to the CPU if CUDA is unavailable). Measure on the hardware you will deploy on.

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError … onnx` | The `onnx` package is not installed | `pip install onnx onnxruntime` (the `edge` extra) |
| ONNX output differs from PyTorch by more than 1e-3 | Model not in `eval()` mode before export, or dropout/batch-norm in training mode | `ModelOptimizer` calls `eval()`; if you export yourself, call `model.eval()` first |
| Export fails on a detector | Detectors take lists of images, not a tensor | Export the backbone or use the framework's own detection export |
| Quantised model much worse | Calibration data not representative, or accuracy-sensitive layers | Calibrate on more, varied chips; keep the float model |
