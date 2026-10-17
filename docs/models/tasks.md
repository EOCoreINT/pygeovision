# Models by task

Pick the **kind of answer** you want and this page sends you to the section that covers it: the models that give it, how to
train one, how to run one, and what goes in and out. Every section is written the same way, so once you have read one you can find
things in the others.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Segmentation
:link: ../segmentation/index
:link-type: doc
**A class for every pixel.** Land cover, crops, water, burned area, buildings, roads. U-Net, DeepLab, SegFormer, UperNet, SAM.
:::

:::{grid-item-card} Classification
:link: ../classification/index
:link-type: doc
**One label per image or chip.** Scene types, crop or no crop, "does this tile contain a port". ResNet, ConvNeXt, Swin, ViT, CLIP.
:::

:::{grid-item-card} Object detection
:link: ../detection/index
:link-type: doc
**A box around every object.** Buildings, vehicles, ships, storage tanks, solar panels. Faster R-CNN, DETR, FCOS, CenterNet.
:::

:::{grid-item-card} Change detection
:link: ../change-detection/index
:link-type: doc
**What changed between two dates.** Methods that need no training, and siamese networks (BIT, DSAMNet, ChangeFormer).
:::

:::{grid-item-card} Foundation models
:link: ../foundation-models/index
:link-type: doc
**Large pretrained networks** used as a starting point: Prithvi-EO, DOFA, DINOv3, SAM, RemoteCLIP, TESSERA, AlphaEarth.
:::

:::{grid-item-card} Vision-language
:link: ../vision-language/index
:link-type: doc
**Words and pictures together.** Captions, questions and text search over thousands of tiles. CLIP, RemoteCLIP, Moondream.
:::

:::{grid-item-card} 3D and LiDAR
:link: ../lidar/index
:link-type: doc
**Point clouds.** Canopy height, building footprints, and networks that label every point (PointNet++, RandLA-Net, KPConv).
:::

:::{grid-item-card} Super-resolution
:link: ../super-resolution/index
:link-type: doc
**Sharper images for people to look at.** SwinIR, with the warning that sharper is not more accurate.
:::

:::{grid-item-card} Time series
:link: ../timeseries/index
:link-type: doc
**Change over many dates.** Trends, seasonality and anomalies from a stack of images.
:::
::::

## What goes in and what comes out

Shapes were measured by building each family with `get_model(..., pretrained=False)` and running it on a small random tensor on
a CPU (`tools/probe_models.py`).

| Task | You give | You get | Start with | Pages |
|---|---|---|---|---|
| Segmentation | `(batch, bands, H, W)` | scores `(batch, classes, H, W)`; the class is the `argmax` | `unet-r50` | [Models](../segmentation/models.md) · [Train](../segmentation/train.md) · [Predict](../segmentation/predict.md) |
| Classification | `(batch, bands, H, W)` | scores `(batch, classes)` | `resnet50` | [Models](../classification/models.md) · [Train](../classification/train.md) · [Predict](../classification/predict.md) |
| Detection | a **list** of `(bands, H, W)` tensors | a list of dictionaries with `boxes`, `labels`, `scores` | `faster-rcnn-r50` | [Models](../detection/models.md) · [Train](../detection/train.md) · [Predict](../detection/predict.md) |
| Change detection | two tensors `(batch, bands, H, W)` | scores `(batch, classes, H, W)` | `bit-r50` | [Models](../change-detection/models.md) · [Train](../change-detection/train.md) · [Predict](../change-detection/predict.md) |
| Point clouds | `(batch, channels, points)` | scores `(batch, classes, points)` | `pointnet2-ssg` | [Models](../lidar/models.md) · [Classify points](../lidar/classify.md) |
| Super-resolution | `(batch, bands, H, W)` | a dictionary; `["reconstruction"]` is `(batch, bands, 4H, 4W)` | `swinir-m` | [Models](../super-resolution/models.md) · [Run](../super-resolution/run.md) |
| Vision-language | a file path and a text | a caption, an answer, or scores | `openclip-b32`, `moondream2` | [Models](../vision-language/models.md) · [CLIP](../vision-language/clip.md) · [Captions](../vision-language/captions.md) |
| Foundation | an image | features or an embedding, not a task answer | `prithvi-300m`, `dofa-base` | [Registry](../foundation-models/models.md) · [Overview](../foundation-models/index.md) |

## Which section do I need?

| You want to know… | Section |
|---|---|
| What is *at each pixel* (a map) | [Segmentation](../segmentation/index.md) |
| What *kind of scene* this tile is | [Classification](../classification/index.md) |
| *How many* of something and *where* each one is | [Detection](../detection/index.md) |
| *Where did it change* between two dates | [Change detection](../change-detection/index.md) |
| The *height* of trees or the shape of buildings from lidar | [3D and LiDAR](../lidar/index.md) |
| Which tiles *look like* a description | [Vision-language](../vision-language/index.md) |
| A good starting point when you have *few labels* | [Foundation models](../foundation-models/index.md) |

```{admonition} Models without training, and models that need it
:class: tip
Some commands run **with no trained weights** because they use a method, not a network: `ai segment water` (NDWI), `ai change`
(change vector analysis), `ai cloud-mask`, `ai classify land-cover` (a published reference map) and `ai classify scene` (CLIP,
which downloads public weights). Everything that runs *your* network (`custom`, `infer`, `--method changeformer|bit|dsamnet`)
requires `--checkpoint`. [Choosing a model](choosing.md) walks through the decision.
```

## Beyond the registry

| Page | What it covers |
|---|---|
| [Choosing a model](choosing.md) | Do you need a model at all? A step-by-step decision |
| [Model catalogue](catalog.md) | All 79 entries in one table |
| [How the registries relate](registry.md) | `get_model`, `pygeovision models`, the zoo and the legacy hub, and how to add your own |
| [Models that are listed but not available](unavailable.md) | Names you may look for, and why they cannot be built |

## Related

- [Training a model](../training/index.md), [Inference](../inference/index.md), [Experiments](../experiments/index.md).
