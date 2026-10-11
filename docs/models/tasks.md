# Models by task

For each kind of model this page says **what it is for**, **what goes in and what comes out**, **how to use it** and
**what to do when it misbehaves**. Shapes and outputs below were produced by running the code (small synthetic
rasters, CPU), so they are what you will see.

| Task | Jump to |
|---|---|
| A class for every pixel | [Segmentation](#segmentation) |
| One label per image | [Scene classification](#scene-classification) |
| Boxes around objects | [Object detection](#object-detection) |
| What changed between two dates | [Change detection](#change-detection) |
| LiDAR points | [Point clouds](#point-clouds) |
| Sharper images | [Super-resolution](#super-resolution) |
| Captions, questions, text search | [Vision-language](#vision-language-models) |

```{admonition} Models without training, and models that need it
:class: tip
Some commands here run **with no trained weights** because they use a method, not a network: `ai segment water`
(NDWI), `ai change` (change vector analysis), `ai cloud-mask`, `ai classify land-cover` (a published reference map) and
`ai classify scene` (CLIP, which downloads public weights). Everything that runs *your* network (`custom`, `infer`,
`--method changeformer|bit|dsamnet`) requires `--checkpoint`.
```

## Segmentation

**Why.** Segmentation answers "what is at every pixel?". It produces the maps most people want: land cover, crop type,
water, burned area, buildings.

**Input and output.** A tensor `(batch, bands, H, W)` goes in; scores `(batch, classes, H, W)` come out. The class of
each pixel is the one with the highest score (`argmax`). Verified with 4 bands and 3 classes on 64 × 64 chips:

| Model | Input | Output |
|---|---|---|
| `unet-r50` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |
| `segformer-b0` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |
| `fcn-r50` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |

All segmentation models return scores at the **input resolution** (SegFormer's internal quarter-resolution output is
resized for you), so the prediction lines up with the raster.

**How to use it.**

```python
import torch
from pygeovision.models import get_model

model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
scores = model.eval()(torch.rand(2, 4, 64, 64))
labels = scores.argmax(dim=1)            # (2, 64, 64) integer class per pixel
```

Train it with [`GeoTrainer`](../training/index.md) and run it over a scene with
[tiled inference](../inference/index.md):

```bash
pygeovision infer predict scene.tif --model unet_resnet50 --checkpoint runs/best.pth -c 3 -o prediction.tif
```

**Without training:**

| Command | Method | Needs |
|---|---|---|
| `pygeovision ai segment water -i stack.tif -o water.tif` | NDWI = (green − NIR) / (green + NIR) > 0 | A blue-green-red-NIR stack (green = band 2, NIR = band 4). Other layouts: `--green-band N --nir-band M` |
| `pygeovision ai segment sam -i image.tif -o masks.tif` | Segment Anything, automatic masks | The SAM weights (downloaded on first use) |
| `pygeovision ai segment buildings -i image.tif -o b.tif` | SAM automatic masks used as building candidates | SAM weights. Candidate shapes, not a validated building map |

**Expected output of `ai segment water`.** A single-band `uint8` GeoTIFF on the input grid: `1` = water, `0` = not
water. On a 4-band synthetic scene:

```text
Running native water segmentation on scene.tif...
✓ Segmentation complete → water.tif
```

An image with fewer than four bands needs the band numbers, and the command stops with a message instead of guessing
(an RGB image would otherwise yield "red minus green", which is not water):

```text
Error: scene.tif has 3 band(s). Water mapping by NDWI needs a green and a near-infrared band: pass green_band= and nir_band= ...
```

For real water mapping use the [`water_bodies`](../pipelines/catalog/water_bodies.md) pipeline: it corrects the
radiometry, masks cloud and reports areas. The command above is a quick tool for imagery you already prepared.

**What to do when…**

| Symptom | Cause | Fix |
|---|---|---|
| Output is one class everywhere | Untrained or badly trained model, or class imbalance | Check validation IoU **per class**; use a [balanced loss](../training/losses.md) |
| Blocky seams between tiles | Tiles predicted independently | Increase `overlap`; keep `blend="gaussian"` ([Inference](../inference/index.md#tiles-overlap-and-seams)) |
| Mask is shifted from the imagery | Never from the model: the output copies the input's grid. Check the input CRS | [Core concepts](../getting-started/concepts.md) |
| `size mismatch` loading a checkpoint | Architecture, classes or bands differ from training | [How the registries relate](registry.md) |

## Scene classification

**Why.** To label a whole image or chip ("this tile is forest"), rather than every pixel.

**Supervised classifiers** (`resnet50`, `convnext-t`, `swin-t`, `vit-b16`, `dinov2-b`, …) take `(batch, bands, H, W)`
and return `(batch, classes)` scores:

| Model | Input | Output |
|---|---|---|
| `resnet50` (5 classes, 4 bands) | `(2, 4, 64, 64)` | `(2, 5)` |
| `vit-b16` (5 classes, 4 bands) | `(1, 4, 224, 224)` | `(1, 5)` |

ViT and Swin need the chip size they were built for: pass `img_size=` to `get_model` when yours differs.

**Zero-shot, with no training**: CLIP compares an image with text descriptions.

```bash
pygeovision ai classify scene --input tile.tif --classes "forest,water,urban,agriculture"
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
scores = client.classification.scene("tile.tif", categories=["forest", "water", "urban", "agriculture"])
# {'forest': 0.71, 'agriculture': 0.18, 'urban': 0.07, 'water': 0.04}   one probability per category, sums to 1
```

The numbers above only illustrate the shape of the result: it is a dictionary from each category you supplied to a
probability. Zero-shot CLIP is a convenient first look. It was trained on photographs, so it is weakest on unusual
sensors and bands; for decisions, train a classifier on your own labelled chips or use `remoteclip-b32`/`-l14`
(remote-sensing-tuned, see [Foundation models](foundation.md)).

**Land cover from a reference map**, not a model:
`pygeovision ai classify land-cover --input s2.tif --output lc.tif` downloads the ESA WorldCover map for the image's
extent (in Python, `client.classification.land_cover(path, year=2021)` chooses the year). It is a published product at 10 m, appropriate for
labels and context; it is not a classification of *your* image.

## Object detection

**Why.** To find and count objects (vehicles, ships, tanks, solar arrays) and locate each with a box.

**Input and output.** torchvision-style detectors take a *list* of image tensors (sizes may differ) and return a list
of dictionaries, one per image:

```python
import torch
from pygeovision.models import get_model

model = get_model("faster-rcnn-r50", num_classes=2, in_channels=4, pretrained=False).eval()
with torch.no_grad():
    out = model([torch.rand(4, 128, 128)])
print({k: tuple(v.shape) for k, v in out[0].items()})
# {'boxes': (100, 4), 'labels': (100,), 'scores': (100,)}
```

`boxes` are `(x1, y1, x2, y2)` in pixels, `scores` are confidences from 0 to 1, and `labels` are class numbers. Keep
boxes whose score exceeds a threshold of your choosing (0.5 is a common start). An untrained model emits up to 100
meaningless boxes; that is expected.

**Training data format.** Each sample is `(image_tensor, target)` with

```python
target = {"boxes": torch.tensor([[x1, y1, x2, y2], ...], dtype=torch.float32),
          "labels": torch.tensor([1, 1, ...])}      # object classes are numbered from 1; 0 is background
```

and `GeoTrainer(model, TrainingConfig(task="detection", ...))`. Validation reports `val_map50` and `val_map50_95`.
Verified: two epochs on synthetic boxes train, validate and write checkpoints. Use `num_classes` = the number of
object classes when building the model (background is added for you).

**YOLO commands without a checkpoint.** `pygeovision ai detect generic|ships|cars` use a YOLO model pretrained on the
COCO photo dataset: "ships" means COCO's *boat* class and "cars" means *car* and *truck*. COCO is made of ground-level
photographs, so these models **often miss or mislabel objects in overhead imagery**. Treat them as a demonstration. For
real work, fine-tune YOLO or Faster R-CNN on overhead labels and run it with `ai detect custom --model … --checkpoint …`,
or call `GeoYOLO(weights="your.pt").detect_tiled(...)` for large rasters, which returns boxes in pixel, map and
longitude/latitude coordinates and can write GeoJSON.

:::{admonition} Detection checkpoints and `infer predict`
:class: warning
`pygeovision infer predict` and tiled inference expect a model that returns a class-score *raster*. Detectors (Faster
R-CNN, FCOS, RT-DETR, YOLO) return boxes and are **not compatible** with it. Run detectors with their own interface as
shown above, or with `GeoYOLO.detect_tiled`.
:::

**What to do when…**

| Symptom | Cause | Fix |
|---|---|---|
| No detections | Score threshold too high, objects too small for the input size, or untrained model | Lower the threshold; train on chips where objects are at least ~10 px; check `val_map50` |
| Many false boxes | Too low a threshold or too little negative data | Raise the threshold; add chips without objects |
| `val_map50` is 0.0 every epoch | The model is not learning yet, or labels use class `0` | Labels must start at 1; train longer; check boxes are inside the image |
| Boxes in the wrong place on the map | Pixel boxes not converted with the raster's transform | Use `GeoYOLO.detect_tiled`, which does it for you |

## Change detection

**Why.** To find where something changed between two dates.

**Without training**: `spectral_diff` (change vector analysis). For every pixel it measures the length of the
band-difference vector, then splits "changed" from "unchanged" with an automatic Otsu threshold.

```bash
pygeovision ai change --before 2020.tif --after 2024.tif -o change.tif
```

```text
Running spectral_diff change detection...
✓ Change detection complete → change.tif
  threshold: 0.001367
  change_fraction: 0.0427
  valid_fraction: 1.0
```

| Output | Meaning |
|---|---|
| `change.tif` | `uint8`; `1` changed, `0` unchanged, `255` no data (nodata in either image). Tags record `method` and `threshold` |
| `threshold` | The magnitude above which a pixel counts as changed, in the rasters' own units |
| `change_fraction` | Share of **valid** pixels that changed |
| `valid_fraction` | Share of pixels valid in both images |

The two rasters must have the same shape and grid; bring them onto one grid first (`pygeovision preprocess clip` to a common bounding box, then `resample` to the same pixel size). On a scene that
did not change, Otsu still splits the noise into two groups and reports a change; pass `--min-magnitude` (an absolute
floor in the raster's units) to prevent that.

**Networks (need your trained checkpoint).**

| Method | What it is | Notes |
|---|---|---|
| `bit` | Bitemporal Image Transformer. Encoder `resnet18` (CLI default) or `resnet50` | Registry: `bit-r50` is the ResNet-50 variant. Run it with `--backbone resnet50` |
| `dsamnet` | Deeply supervised attention network | Registry `dsamnet` is the `resnet18` variant |
| `changeformer` | A compact siamese cross-attention network (~0.8 M parameters). **Not** the published transformer ChangeFormer | One fixed architecture; no `--backbone` |

Two-image networks take `model(before, after)` and return `(batch, classes, H, W)`:

| Model | Inputs | Output |
|---|---|---|
| `bit-r50`, `dsamnet`, `changeformer-mit-b0` | two `(2, 4, 64, 64)` tensors | `(2, 2, 64, 64)` |

`changeformer-mit-b0` also accepts one tensor with the two images stacked on the channel axis, `[before | after]`, which
is the form `GeoTrainer` can feed. `bit` and `dsamnet` need two arguments, so wrap them for training:

```python
import torch.nn as nn

class Stacked(nn.Module):
    """GeoTrainer feeds one tensor; split it into the before and after images."""
    def __init__(self, net, bands):
        super().__init__()
        self.net, self.bands = net, bands
    def forward(self, x):
        return self.net(x[:, :self.bands], x[:, self.bands:])

net = get_model("bit-r50", num_classes=2, in_channels=4, pretrained=False)
trainer = GeoTrainer(Stacked(net, 4), cfg)        # dataset items: (8-band stack, mask)
trainer.fit(train_ds, val_ds)
torch.save(net.state_dict(), "bit.pth")           # save the inner network so the CLI can load it
```

Save `net.state_dict()`, not the trainer's checkpoint: the trainer's keys carry the wrapper's `net.` prefix, which the
CLI does not expect. Then:

```bash
pygeovision ai change --before a.tif --after b.tif -o change.tif --method bit --backbone resnet50 --checkpoint bit.pth
```

(Verified end to end. The synthetic two-epoch network in this test marked 99.997 % of the scene as changed: an
under-trained network gives a confident-looking map. Always inspect the result and validate it.)

**What to do when…**

| Symptom | Cause | Fix |
|---|---|---|
| `before/after rasters differ in shape` | Different extent or resolution | `preprocess clip` to the same box and `preprocess resample` to the same pixel size |
| Change map is mostly "changed" with `spectral_diff` | Different sun angle, season or sensor between dates | Use dates in the same season; correct radiometry first; set `--min-magnitude` |
| `size mismatch` with `--method bit` | Encoder differs from training | Add `--backbone resnet50` (or `resnet18`) to match |
| Clouds appear as change | Cloud differs between dates | Mask cloud before differencing ([Preprocessing](../data/preprocessing.md)) |

## Point clouds

**Why.** To classify each LiDAR point (ground, vegetation, building, …).

**Input and output.** `(batch, channels, points)` in, `(batch, classes, points)` out, where the first three channels
are x, y, z and any further channels are per-point features such as intensity. With `in_channels=4` (x, y, z and one
feature) and 4 classes, 512 points:

| Model | Input | Output |
|---|---|---|
| `pointnet2-ssg`, `randlanet`, `kpconv`, `ptv3` | `(2, 4, 512)` | `(2, 4, 512)` |

`in_channels` counts xyz plus features. Models are built from scratch; there are no pretrained point-cloud weights.
Use [point-cloud processing](../advanced/index.md) to read LAS/LAZ files and derive canopy-height models.

## Super-resolution

**Why.** To produce a higher-resolution image from a lower-resolution one. The result is *plausible detail*, not
measured detail: do not use it to measure small objects.

`swinir-m` (Swin2SR, 4×) takes `(1, 3, 32, 32)` and returns `{'reconstruction': (1, 3, 128, 128)}`. The pretrained
weights are for **three-band RGB** only; for other band counts build with `pretrained=False` and train, or process
three bands at a time.

## Vision-language models

**Why.** To describe an image in words, answer questions about it, or search a folder of images by text.

```bash
pygeovision vlm caption scene.tif
pygeovision vlm query scene.tif "How many buildings are visible?"
pygeovision vlm search "flooded fields" ./tiles --top-k 5
```

`vlm caption` and `vlm query` use Moondream2 (about 1.8 B parameters); `vlm search` ranks images by CLIP similarity to
your text. All download public weights on first use. Answers are fluent and **can be wrong**, especially for counts and
small objects; never treat a caption as a measurement. See [Foundation models](foundation.md#vision-language).

## Models that are listed but not available

The registry names some models so you can discover them; building them raises `NotImplementedError` and says why.
These are `rf-detr-b/-l` (distributed in Roboflow's own package), `dinov3_segmentor`, `prithvi_burn_scar`,
`changestar-r18`, `changeformer-mit-b4`, `satlas-pretrain`, `chatearthnet` and `lisat-7b`. The
[catalogue](catalog.md) marks them *listed only*. Nothing is substituted for them.
