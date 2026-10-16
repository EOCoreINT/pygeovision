# Segmentation

Segmentation answers **"what is at every pixel?"** It produces the maps most people want: land cover, crop type, water,
burned area, buildings, roads. This section covers the models, how to train one, and how to run one over a scene.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Models
:link: models
:link-type: doc
U-Net, DeepLab, PSPNet, FCN, SegFormer, UperNet, Mask2Former and SAM, with measured sizes and shapes.
:::

:::{grid-item-card} Train one
:link: train
:link-type: doc
Labels, configuration, fine-tuning, reading the log and testing the result.
:::

:::{grid-item-card} Predict
:link: predict
:link-type: doc
The command line, Python, methods that need no training, and tiled inference over a whole scene.
:::
::::

## What goes in and what comes out

A tensor `(batch, bands, H, W)` goes in; scores `(batch, classes, H, W)` come out. The class of each pixel is the one with
the highest score (`argmax`). Verified with 4 bands and 3 classes on 64 × 64 chips:

| Model | Input | Output |
|---|---|---|
| `unet-r50` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |
| `segformer-b0` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |
| `fcn-r50` | `(2, 4, 64, 64)` | `(2, 3, 64, 64)` |

All segmentation models return scores at the **input resolution** (SegFormer's internal quarter-resolution output is
resized for you), so the prediction lines up with the raster pixel for pixel.

```python
import torch
from pygeovision.models import get_model

model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
scores = model.eval()(torch.rand(2, 4, 64, 64))
labels = scores.argmax(dim=1)            # (2, 64, 64) integer class per pixel
```

## The workflow

```{figure} ../_static/img/flow-segmentation.svg
:alt: Segmentation workflow: label, chip, train, check, predict
:width: 100%
:class: pgv-figure

From labels to a map: each step has its own page.
```

1. **Labels**: a raster of class numbers on the same grid as the image ([labelling](../labeling/index.md)).
2. **Chips and a split**: [Preparing training data](../training/data.md).
3. **Model**: [Segmentation models](models.md).
4. **Train**: [Train a segmentation model](train.md).
5. **Predict**: [Predict with a segmentation model](predict.md).

## Do you need to train?

| Situation | What to use |
|---|---|
| Water | `pygeovision ai segment water` (NDWI) or the [`water_bodies`](../pipelines/catalog/water_bodies.md) pipeline. No training |
| A published land-cover map is enough | [`land_cover`](../pipelines/catalog/land_cover.md) (ESA WorldCover). No training |
| Outlines of objects, no class names | SAM ([SAM](../foundation-models/sam.md)). No training |
| Your own classes, your own sensor or region | Train: [Train a segmentation model](train.md) |
| Very few labels, multispectral data | Fine-tune a foundation model ([Prithvi-EO](../foundation-models/prithvi.md)) |

## Pipelines that use segmentation

Pipelines that run **your trained network** are the ones that need `--checkpoint`:
[`crop_monitoring`](../pipelines/catalog/crop_monitoring.md), [`building_footprints`](../pipelines/catalog/building_footprints.md)
(also runs without), [`road_extraction`](../pipelines/catalog/road_extraction.md) and
[`land_cover`](../pipelines/catalog/land_cover.md) with a model name as the `source`.

## Reading a segmentation score

`val_iou` in the training log is the **mean intersection over union** over the classes that occur: for each class, the
overlap of prediction and reference divided by their union. 1.0 is perfect. Because background usually dominates, a mean
can look good while a rare class is missed entirely, so always read the per-class values ([train](train.md#read-the-per-class-values)).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| Output is one class everywhere | Untrained or badly trained model, or class imbalance | Check validation IoU **per class**; use a [balanced loss](../training/losses.md) |
| Blocky seams between tiles | Tiles predicted independently | Increase `overlap`; keep `blend="gaussian"` ([Inference](../inference/tiles.md#tiles-overlap-and-seams)) |
| Mask is shifted from the imagery | Never from the model: the output copies the input's grid. Check the input CRS | [Core concepts](../getting-started/concepts.md) |
| `size mismatch` loading a checkpoint | Architecture, classes or bands differ from training | [How the registries relate](../models/registry.md) |
