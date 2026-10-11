# Inference: running a trained model on imagery

Inference is applying a trained model to new imagery to produce a map. A satellite scene is far too large to hand to a
network in one piece (a 10,000 × 10,000 pixel scene would need tens of gigabytes of working memory), so PyGeoVision
cuts it into overlapping **tiles**, predicts each, and stitches the answers back into one georeferenced raster. This page
explains how that works, which command or class to use, how to feed the model data it understands, and how to handle
nodata, seams, probabilities and detectors.

```{admonition} The three things that decide whether the map is right
:class: important
1. **The model and settings are the ones used in training**: same name, classes, bands ([How the registries relate](../models/registry.md)).
2. **The pixel values are scaled the way they were in training** ([see below](#matching-the-input-to-training)).
3. **You looked at the result** next to the imagery, not just at a success message.
```

## Which entry point

| You want to | Use | Notes |
|---|---|---|
| Predict one scene from the shell | `pygeovision infer predict` | The standard command |
| Predict a folder of scenes | `pygeovision infer batch` | Skips outputs that already exist |
| The same, with `ai` options | `pygeovision ai infer` | Same engine; `--tile-size` instead of `--chip-size` |
| Predict from Python, full control | `TiledInference(...).infer(...)` | Every option below |
| Predict an array you already hold in memory | `TiledInference(...).predict_array(array)` | No stretch, returns probabilities |
| Many scenes in Python | `BatchInferenceEngine` | Same tiling, threads, error reporting |
| Combine several models | `EnsembleInference` | Averages class probabilities |
| Find objects with boxes | See [Detection models](#detection-models) | Not `TiledInference` |

All of these except the box detectors use one engine, `pygeovision.inference.TiledInference`.

## Predicting a scene

```bash
pygeovision infer predict scene.tif \
    --model unet-r50 --checkpoint runs/demo/checkpoints/best.pth \
    --classes 2 --in-channels 4 \
    --chip-size 128 --overlap 16 --blend gaussian \
    --no-normalise \
    --output prediction.tif
```

Expected output (a real run on a 256 × 256 scene):

```text
  Inference: scene.tif
  Model:     unet-r50 | classes=2 | chip=128 | blend=gaussian
  Input:     used as stored (no stretch)
  ✓ Output:  prediction.tif
  ✓ Chips:   9 | Time: 1.0s
```

| Option | Default | Meaning |
|---|---|---|
| `IMAGE_PATH` | required | The GeoTIFF to predict |
| `-m, --model` | `unet_resnet50` | The architecture name you trained ([registries](../models/registry.md)) |
| `--checkpoint` | required | Your trained weights. The registry ships architectures, not trained models, so there is no default |
| `-c, --classes` | 2 | Number of output classes of the checkpoint |
| `--in-channels` | the image's band count | Bands the model was trained on. Give it explicitly |
| `--chip-size` | 512 | Tile edge in pixels. Use the size you trained with, or larger |
| `--overlap` | 64 | Pixels shared by neighbouring tiles ([seams](#tiles-overlap-and-seams)) |
| `--blend` | `gaussian` | How overlapping predictions are combined: `gaussian`, `linear`, `constant` |
| `--device` | auto | `cuda`, `cpu` or `mps`; auto picks CUDA when available |
| `--normalise / --no-normalise` | `--normalise` | Whether to stretch each band to its 2nd–98th percentile first ([see below](#matching-the-input-to-training)) |
| `-o, --output` | `<input>_<model>_pred.tif` | Where to write |

If the checkpoint does not fit the model, loading stops with a message that names the mismatch; nothing is predicted:

```text
Error: checkpoint runs/demo/checkpoints/best.pth does not match 'unet-r50' (3 classes, 4 bands): Error(s) in loading
state_dict for Unet:
	size mismatch for segmentation_head.0.weight: copying a param with shape torch.Size([2, 16, 3, 3]) from checkpoint,
	the shape in current model is torch.Size([3, 16, 3, 3]).
```

### What the output is

A single-band **`uint8` GeoTIFF**, same size, CRS and transform as the input, where each pixel holds the **class
number** (0, 1, …) of the most likely class. The tags record how it was made (`method`, `blend_mode`, `chip_size`,
`overlap`, `tta`).

- **No-data pixels are 255** and the raster declares `nodata=255`. A pixel is "no data" when it is NaN/Inf in any
  band or equals the file's nodata value in every band. If the scene has none, no nodata flag is written. (Class
  numbers 0 to 254 are therefore possible; keep `num_classes` below 255.)
- With `return_probabilities=True` (Python only) the file holds one band per class instead ([below](#probabilities-instead-of-classes)).

The Python call returns a dictionary:

```python
{'success': True, 'output_path': 'prediction.tif', 'n_chips': 9, 'image_size': (256, 256),
 'valid_fraction': 1.0, 'duration_seconds': 0.7, 'chips_per_second': 12.1, 'blend_mode': 'gaussian', 'tta': False}
```

On failure it returns `{'success': False, 'error': '…'}` (for instance when the model does not accept the number of
bands in the file) instead of writing a bad map; the commands turn that into an error and a non-zero exit code.

## Matching the input to training

This is the most important section on the page.

A network is trained on numbers in a particular range. If at prediction time the same scene arrives scaled differently,
the network sees something it has never seen, and the output is wrong in a way that looks plausible.

`TiledInference.infer` and the commands **stretch each band by default**: for every band, the 2nd and 98th percentiles of
the valid pixels *of that scene* become 0 and 1 (`value = (value − p2) / (p98 − p2)`, not clipped). That is right for
a model trained on imagery stretched the same way, and wrong for a model trained on, say, reflectance 0–1 used as
stored.

A real demonstration. A model scored `val_iou` 0.98 in training on chips used as stored. Predicting the validation
scene gave:

| Prediction | Pixels agreeing with the truth | Pixels predicted as the target class |
|---|---:|---:|
| default (`--normalise`: stretched first) | 82.1% | 64 (of 11,781 true) |
| `--no-normalise` (as in training) | **99.6%** | 11,927 |

With the stretch applied, the model found almost nothing; the 82% "agreement" is just the background share of the
scene. Always compare against what you know.

### Rules

1. **Decide the scaling once** and apply it identically in training and prediction. The options are in
   [Preparing training data](../training/data.md#scaling-the-pixel-values).
2. If you trained on **fixed-scale data** (reflectance 0–1, z-scored indices, decibels): predict with
   `--no-normalise` (command) or `normalise=False` (Python), after giving the scene the same scaling, for example with
   `client.prepare_for_ai(..., normalise="scale_factor")`.
3. If you trained on **per-scene percentile-stretched** data: use the default, and make sure training imagery was
   stretched the same way (the same two percentiles, per band, per scene).
4. If you cannot tell how a checkpoint was trained, you cannot trust its maps. Keep a note beside every checkpoint.

### Using a stack you have already prepared

`predict_array` applies **no** stretch and returns the probabilities, so a feature stack prepared for training reaches the
model exactly as it was trained:

```python
from pygeovision.inference import TiledInference

engine = TiledInference(model, chip_size=256, overlap=32, num_classes=5, device="cpu")
probabilities = engine.predict_array(stack)          # stack: (bands, H, W) float32  ->  (5, H, W) float32
labels = probabilities.argmax(axis=0)
```

`nan_value=0.0` (default) replaces NaN and Inf before prediction, as training code usually does; pass `None` to leave
them.

## Python API

```python
from pygeovision.models import get_model
from pygeovision.inference import TiledInference
import torch

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("best.pth", weights_only=True)["model_state_dict"])   # strict by default

engine = TiledInference(model, chip_size=256, overlap=32, num_classes=2, blend_mode="gaussian")
result = engine.infer("scene.tif", "prediction.tif", normalise=False)
```

Constructor:

| Argument | Default | Meaning |
|---|---|---|
| `model` | required | A `torch.nn.Module` that maps `(B, bands, H, W)` to class scores `(B, classes, H, W)` (or an object with `.logits`, as Hugging Face models return) |
| `chip_size` | 512 | Tile edge in pixels |
| `overlap` | 128 | Pixels shared between neighbours; must be less than `chip_size` |
| `blend_mode` | `"gaussian"` | `gaussian`, `linear` or `constant` ([seams](#tiles-overlap-and-seams)) |
| `sigma_ratio` | 0.25 | Width of the Gaussian window as a fraction of the chip |
| `batch_tiles` | 4 | Tiles collected before processing; mostly a bookkeeping value (tiles are predicted one by one) |
| `device` | auto | `"cuda"`, `"cpu"`, `"mps"` |
| `num_classes` | 2 | Number of output channels to accumulate |
| `activation` | `"softmax"` | `softmax` (one class per pixel), `sigmoid` (independent classes), `none` (raw scores) |
| `tta` | `False` | Test-time augmentation: average the prediction of the chip and its three flips. About 4× slower |
| `half_precision` | `True` | Use half precision **on a CUDA GPU** only; no effect on CPU |
| `tolerate_chip_failures` | `False` | If a chip fails, raise (default). `True` fills it with zeros and goes on, which silently corrupts the map; do not use it to hide a mismatch |

`infer` arguments:

| Argument | Default | Meaning |
|---|---|---|
| `image_path`, `output_path` | | Input and output GeoTIFF |
| `return_probabilities` | `False` | Write one band per class (see below) instead of the class number |
| `band_selection` | all | 1-based band numbers to feed the model, for a file with more bands than the model takes |
| `normalise` | `True` | Percentile stretch ([rules](#rules)) |
| `nodata_value` | the file's nodata | Value that means "no data" |

### Probabilities instead of classes

```python
engine.infer("scene.tif", "prob.tif", return_probabilities=True, normalise=False)
```

The output has **one band per class**, `uint8`, with the probability scaled to 0–255 (a pixel at 249 is about 98%
likely). Use it to apply your own threshold: for a rare class a cut at 0.3 instead of the implicit "most likely" finds
more targets at the cost of more false alarms. No-data pixels are 0 in every band.

### Memory

`infer` reads the scene into memory and keeps accumulators for the whole output, so memory grows with the scene:
roughly `bands × 4 + classes × 8 + 8` bytes per pixel. For a 10,000 × 10,000 pixel, 4-band, 2-class scene that is about 4 GB.
Check before starting:

```python
engine.estimate_memory(10000, 10000)
# {'n_chips': 8100, 'batch_gpu_mb': 0.5, 'accumulation_ram_mb': 1525.9, 'estimated_total_mb': 1526.4}
```

This estimate covers the output accumulators only (and the GPU figure counts the tile scores, not the network's
internal activations). It does not include the input image or the model. If it is too much, cut the scene into pieces
first (`pygeovision preprocess clip`), predict each, and mosaic. `StreamingInference` lowers the model's memory but
**still reads the whole scene**; it is not out-of-core.

## Tiles, overlap and seams

A network sees less context at the edge of a tile and is less reliable there. Where neighbouring tiles meet with no
overlap, the two sides can disagree and leave a visible line. The remedy is to make tiles overlap and to **weight**
each prediction by how far the pixel is from the tile's edge, so the confident centre of every tile counts most.

A measurement (a trained U-Net on a 256 × 256 scene; "differs" is the share of pixels whose class differs from the
prediction made on the whole scene in one pass):

| Chip | Overlap | Blend | Differs from whole-scene prediction | Time |
|---:|---:|---|---:|---:|
| 128 | 0 | constant (no blending) | 0.31% | 0.4 s |
| 128 | 0 | gaussian | 0.31% | 0.3 s |
| 128 | 16 | gaussian | **0.03%** | 0.9 s |
| 128 | 64 | gaussian | 0.03% | 1.9 s |
| 128 | 16 | linear | 0.03% | 0.8 s |
| 128 | 16 | constant | 0.04% | 0.7 s |
| 64 | 0 | constant | 0.92% | 0.8 s |
| 64 | 16 | gaussian | 0.14% | 1.9 s |
| 64 | 32 | gaussian | 0.12% | 3.2 s |

What this shows, and does not:

- A **small overlap does most of the work** (0.31% → 0.03%); a large one only costs time.
- **Smaller tiles disagree more** (0.92% at 64 pixels with no overlap) because each sees less context.
- This is one small, synthetic scene. On real imagery with fine structure (roads, small buildings), seams matter more.

Guidance:

| Choose | Rule of thumb |
|---|---|
| `chip_size` | At least the size you trained on, as large as memory allows (256–1024). Larger tiles see more context |
| `overlap` | About 10–25% of `chip_size` (32 for 256, 64 for 512). More gives diminishing returns |
| `blend` | `gaussian` (default) unless you have a reason |

Notes: tiles on the right and bottom edges are padded with zeros so the model always sees a full-size chip, and their
weight is set so they affect only the pixels they cover. Every pixel gets a prediction, including the outermost ones.

### Test-time augmentation

`tta=True` predicts each tile four times (original, flipped horizontally, flipped vertically, both) and averages. It
smooths noisy predictions and typically adds a little accuracy, at about four times the cost. In the same test it
changed 0.28% of pixels relative to a single pass. Use it for final maps, not for exploring.

## Predicting many scenes

```bash
pygeovision infer batch scenes_in/ predictions_out/ -m unet-r50 --checkpoint best.pth -c 2 --in-channels 4 --no-normalise
```

```text
  Batch inference: scenes_in → predictions_out
  ✓ 2/2 succeeded
  Time: 1.9s | 1.04 fps
```

- Outputs go to `predictions_out/` with the **same file names** (subfolders are mirrored).
- **Files whose output already exists are skipped**, so an interrupted batch can be restarted. The summary then says
  `2 skipped (output already exists; --overwrite to redo)`; use `--overwrite` to predict again. Exit code is 0 when
  nothing failed.
- A file that fails (for instance the wrong number of bands) is listed with its error and counted as failed; the exit
  code is 1.
- Options: `--workers` (threads; default 2), `--pattern` (default `*.tif`). The CLI uses chip size 512 and overlap 64;
  use Python for other values:

```python
from pygeovision.inference import BatchInferenceEngine

engine = BatchInferenceEngine(model, device="cuda", n_workers=2, chip_size=256, overlap=32,
                              num_classes=2, normalise=False)
summary = engine.run_directory("scenes_in/", "predictions_out/", pattern="*.tif", skip_existing=True)
print(summary["n_success"], summary["n_failed"], summary["failed"])
```

`run_directory` returns `n_success`, `n_failed`, `failed` (a list of `{path, error}`), `skipped`, `total_time_s`,
`throughput_fps`. With `on_error="abort"` the first failure stops the run.

## Combining several models

```python
from pygeovision.inference.stream import EnsembleInference

ens = EnsembleInference([model_a, model_b, model_c], weights=[1, 1, 2], fusion="mean", num_classes=2,
                        normalise=False)
ens.infer("scene.tif", "ensemble.tif")
```

Each model predicts the scene's probabilities, which are weighted (`weights` are relative importances), combined by
`fusion` (`"mean"` or `"max"`; anything else raises `ValueError`), and the most likely class is written. Use it when
you have several models trained on different data or seeds and want a steadier map. It costs one full prediction per
model. Intermediate files are removed. All models must share class meanings and input scaling.

## Detection models

Object detectors (Faster R-CNN, FCOS, YOLO, DETR) answer a different question: not "which class is each pixel" but "where
are the objects". Their output is a list of boxes with scores, not a score per pixel. **`TiledInference` and
`pygeovision infer predict` do not run them**; they expect a model that maps an image tensor to class scores per pixel.
A detector passed to them fails with a clear error on the first chip.

| Task | Use |
|---|---|
| Ships or cars with the generic pretrained YOLO | `pygeovision ai detect ships --input port.tif --output ships.geojson` (needs `pip install ultralytics`). Writes **GeoJSON** boxes; the command refuses a `.tif` path and reports the detector's error if there is one |
| Your own YOLO weights | `GeoYOLO(weights="best.pt").detect_tiled("ortho.tif", tile_size=640, output_path="found.geojson")` |
| A torchvision detector you trained with `GeoTrainer(task="detection")` | Run it chip by chip yourself (below) |

```{admonition} "ai detect custom" is not for box detectors
:class: warning
`pygeovision ai detect custom` runs your model through the per-pixel engine, so it suits models that output a **score
map per class** (for example a segmentation model used as a detector of blobs). It does not run a torchvision detector.
```

Running a trained torchvision detector on a tile and getting georeferenced boxes:

```python
import rasterio, torch
from rasterio.windows import Window
from pygeovision.models import get_model

model = get_model("faster-rcnn-r50", num_classes=1, in_channels=4, pretrained=False, min_size=128, max_size=128)
model.load_state_dict(torch.load("runs/det/checkpoints/best.pth", weights_only=True)["model_state_dict"]); model.eval()

with rasterio.open("scene.tif") as src:
    chip = torch.from_numpy(src.read(window=Window(0, 0, 128, 128)).astype("float32"))
    with torch.no_grad():
        out = model([chip])[0]          # {'boxes': (N, 4) x1,y1,x2,y2 in pixels, 'labels': (N,), 'scores': (N,)}
    for (x1, y1, x2, y2), score in zip(out["boxes"].tolist(), out["scores"].tolist()):
        if score < 0.5:
            continue
        west, north = src.transform * (x1, y1)
        east, south = src.transform * (x2, y2)
        print(f"box ({west:.1f}, {south:.1f}, {east:.1f}, {north:.1f})  score {score:.2f}")
```

`out` holds `boxes` `(N, 4)`, `labels` `(N,)` and `scores` `(N,)` with N up to 100 candidates; most of them have low scores, which is why the
threshold is applied. (A detector trained for two epochs on toy data keeps nothing above 0.5; with a properly trained model
the loop prints one line per object.)

Loop this over a grid of windows (with overlap, then remove duplicates with `torchvision.ops.nms`) for a large scene.
Apply the same scaling to the chip as in training.

## The other inference engine

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

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| The map is almost all one class though validation was good | Scaling mismatch | `--no-normalise` if you trained on fixed-scale data; otherwise stretch training data the same way ([rules](#rules)) |
| `size mismatch` / `Missing key(s)` | Model name, `--classes` or `--in-channels` differ from training | Use the training values |
| `Model does not accept N input band(s)` | The file has more bands than the model | `--in-channels` to match training, or `band_selection=[…]` in Python |
| `Could not build 'X'` | The name is in neither registry | `pygeovision models list`; [registries](../models/registry.md) |
| Grid lines visible in the map | No overlap, or tiles too small | `--overlap` 10–25% of `--chip-size`, `gaussian` blending, larger `--chip-size` |
| Out of memory | Scene too large for RAM or GPU | Smaller `--chip-size` (GPU); clip the scene into pieces (RAM); `device="cpu"` |
| Very slow on CPU | Large model, large scene | A GPU; smaller `--overlap`; `tta=False`; export to ONNX ([Advanced training](../training/advanced.md#exporting-and-shrinking-a-model)) |
| Pixels of value 255 in the output | They are no-data (NaN or the nodata value in the input) | Expected; mask them in your GIS or by `nodata=255` |
| The whole map is class 0 in one corner | Padded edge or no-data region in the input | Check the input there; with the current engine every real pixel is predicted |
| Chip prediction failed … Detection models … not compatible | A detector or change-detection model was passed to `TiledInference` | See [Detection models](#detection-models); use `ai change` for change models |
| `✓ 0/0 succeeded` in the old output | All outputs existed and were skipped | Now reported as skipped; `--overwrite` to redo |
| Different result each run | Dropout or batch-norm in training mode | The engine calls `eval()`; if you pass your own already-wrapped model, check it is in eval mode |
