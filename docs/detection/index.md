# Object detection

Detection finds **discrete objects** (vehicles, ships, tanks, solar arrays, buildings) and gives each a box, a class and a
score. It is the tool for counting and locating things. For continuous surfaces such as land cover use
[segmentation](../segmentation/index.md); for one label per tile use [classification](../classification/index.md).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Models
:link: models
:link-type: doc
Faster R-CNN, Mask R-CNN, FCOS, CenterNet, DETR, RT-DETR and Grounding DINO.
:::

:::{grid-item-card} Train one
:link: train
:link-type: doc
The dataset format, `GeoTrainer(task="detection")`, mAP and what to check.
:::

:::{grid-item-card} Predict
:link: predict
:link-type: doc
Chips, overlap, duplicate removal and georeferenced GeoJSON; YOLO commands.
:::
::::

## What goes in and what comes out

torchvision-style detectors take a *list* of image tensors (sizes may differ) and return a list of dictionaries, one per
image:

```python
import torch
from pygeovision.models import get_model

model = get_model("faster-rcnn-r50", num_classes=2, in_channels=4, pretrained=False).eval()
with torch.no_grad():
    out = model([torch.rand(4, 128, 128)])
print({k: tuple(v.shape) for k, v in out[0].items()})
# {'boxes': (100, 4), 'labels': (100,), 'scores': (100,)}
```

`boxes` are `(x1, y1, x2, y2)` in pixels, `scores` are confidences from 0 to 1, and `labels` are class numbers. Keep boxes
whose score exceeds a threshold of your choosing (0.5 is a common start). An untrained model emits up to 100 meaningless
boxes; that is expected. DETR-style models are different (a tensor in, `logits` and `pred_boxes` out); see
[Detection models](models.md#input-and-output).

## The workflow

```{figure} ../_static/img/flow-detection.svg
:alt: Detection workflow: boxes, chips, train, score, detect
:width: 100%
:class: pgv-figure

From box labels to georeferenced objects.
```

## Choosing a route

| You have | Use |
|---|---|
| Nothing, and a quick demonstration is enough | `pygeovision ai detect generic\|ships\|cars` (YOLO pretrained on photographs; often misses objects in overhead imagery) |
| Box labels for your objects | Train Faster R-CNN, FCOS or CenterNet: [Train a detector](train.md) |
| Objects you cannot label | Open-vocabulary detection (Grounding DINO, needs a network) |
| Building footprints for a place | The [`building_footprints`](../pipelines/catalog/building_footprints.md) pipeline |
| Solar panels, with your trained weights | [`solar_detection`](../pipelines/catalog/solar_detection.md) |

## YOLO commands without a checkpoint

`pygeovision ai detect generic|ships|cars` use a YOLO model pretrained on the COCO photo dataset: "ships" means COCO's
*boat* class and "cars" means *car* and *truck*. COCO is made of ground-level photographs, so these models **often miss or
mislabel objects in overhead imagery**. Treat them as a demonstration. For real work, fine-tune YOLO or Faster R-CNN on
overhead labels and run it with `ai detect custom --model … --checkpoint …`, or call
`GeoYOLO(weights="your.pt").detect_tiled(...)` for large rasters, which returns boxes in pixel, map and longitude/latitude
coordinates and can write GeoJSON.

:::{admonition} Detection checkpoints and `infer predict`
:class: warning
`pygeovision infer predict` and tiled inference expect a model that returns a class-score *raster*. Detectors (Faster
R-CNN, FCOS, RT-DETR, YOLO) return boxes and are **not compatible** with it. Run detectors with their own interface as
shown on the [predict page](predict.md), or with `GeoYOLO.detect_tiled`.
:::

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| No detections | Score threshold too high, objects too small for the input size, or untrained model | Lower the threshold; train on chips where objects are at least ~10 px; check `val_map50` |
| Many false boxes | Too low a threshold or too little negative data | Raise the threshold; add chips without objects |
| `val_map50` is 0.0 every epoch | The model is not learning yet, or labels use class `0` | Labels must start at 1; train longer; check boxes are inside the image |
| Boxes in the wrong place on the map | Pixel boxes not converted with the raster's transform | Use `GeoYOLO.detect_tiled`, which does it for you, or the transform in [predict](predict.md) |

```{toctree}
:hidden:
:maxdepth: 1

models
train
predict
```
