# Foundation models

A *foundation model* is a large network pretrained on huge amounts of unlabelled or loosely labelled imagery. It does not solve your task
directly. It gives you a strong **starting point**: either *features* (a vector that describes an image or a patch, usable for search,
clustering or a small classifier) or *weights to fine-tune* with far fewer labels than training from scratch.

```{figure} ../_static/img/tree-foundation-chooser.svg
:alt: Choosing a foundation model: with no labels, CLIP ranks tiles by words, SAM outlines objects, DINOv3 gives features to cluster; with a few hundred labels, Prithvi suits six HLS bands, DOFA any sensor, DINOv3 plus a head ordinary colour images, and radar must be mapped to six bands first; ready-made yearly vectors come from TESSERA or AlphaEarth
:width: 100%
:class: pgv-figure

Start from what you have, not from the model's name.
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Prithvi-EO
:link: prithvi
:link-type: doc
**NASA/IBM, six HLS bands.** Features, a trainable segmentation head, multi-date change screening, and rule-based fallbacks that say so.
:::

:::{grid-item-card} DOFA
:link: dofa
:link-type: doc
**One model for any sensor or band set.** Sentinel-2, NAIP, Sentinel-1: pass the wavelengths and it adapts.
:::

:::{grid-item-card} DINOv3
:link: dinov3
:link-type: doc
**General visual features**, dense patch features, attention maps. Gated download.
:::

:::{grid-item-card} CLIP and RemoteCLIP
:link: clip
:link-type: doc
**Words and images in one space.** Zero-shot labels and text search over tiles.
:::

:::{grid-item-card} SAM and SAM 2
:link: sam
:link-type: doc
**Outlines of objects from a prompt.** Labelling help; masks have no class names.
:::

:::{grid-item-card} Embeddings
:link: embeddings
:link-type: doc
**TESSERA and AlphaEarth**: a vector per pixel, ready made. Classify from a few points.
:::

:::{grid-item-card} Radar
:link: radar
:link-type: doc
**Using optical models with Sentinel-1.** Normalise, map to six channels, validate. Zero-shot is a threshold.
:::

:::{grid-item-card} The registry
:link: models
:link-type: doc
**Every registered entry**, with measured parameter counts and build status.
:::
::::

## The models side by side

| Model | What it learned from | Use it for | Output | Needs |
|---|---|---|---|---|
| [Prithvi-EO](prithvi.md#prithvi-eo) | NASA/IBM: Harmonized Landsat-Sentinel-2 (HLS), 6 bands. 100 M, 300 M or 600 M parameters | Fine-tuned segmentation (flood, burn, crop, land cover); change in feature space | A feature vector, a patch grid of features, a head you train | Hub download, `einops`, `timm` |
| [DOFA](dofa.md#dofa) | Many sensors, any band set. 86 M (base) or 304 M (large) | One model for Sentinel-2, NAIP, Sentinel-1… | One vector per image | `torchgeo` |
| [DINOv3](dinov3.md#dinov3) | Meta: web images (LVD) or satellite images (SAT-493M) | General visual features, dense features, attention | Global vector, patch grid, attention maps | **Gated** Hub access |
| [CLIP / RemoteCLIP](clip.md#clip-and-remoteclip) | Image-text pairs. 151 M (B/32) or 428 M (L/14) | Zero-shot labels, text search | Image and text vectors | `transformers` or `open_clip_torch` |
| [SAM / SAM2](sam.md#sam-and-sam2) | Promptable segmentation. 94 M (base) to 636 M (huge) | Outlining objects from points or boxes | Masks without class names | Hub download |
| [TESSERA](embeddings.md#tessera) | Cambridge: a year of Sentinel-1/2 | Per-pixel embeddings, ready made | 128 channels per pixel, 10 m | `geotessera` |
| [AlphaEarth](embeddings.md#alphaearth) | Google DeepMind: multi-sensor annual | Per-pixel embeddings, ready made | 64 channels per pixel, 10 m | Earth Engine account |

Parameter counts are the registry's figures ([registry page](models.md)); the real count of a downloaded checkpoint can differ slightly.

## What a foundation model is not

```{admonition} A foundation model is not a map
:class: warning
Feeding an image to Prithvi gives you *features*. To get "flooded / not flooded" you must put a head on top and train it
on labels, or use another method. PyGeoVision follows this strictly: where no trained head is supplied the task falls
back to a documented spectral rule and says so in the result (`result["method"]`), or it refuses, rather than returning random class maps.
```

The same rule runs through the package, and it is the reason several things on these pages raise an error instead of "working":

| If this happens | PyGeoVision does | Not this |
|---|---|---|
| The weights cannot be downloaded | Raises `ModelLoadError` naming the file | Returns random weights that look like a model |
| You ask for a task with no trained head | Runs a named spectral rule (`method == "spectral"`), or raises where no honest rule exists | Pushes features through an untrained layer |
| A repository is gated | Says you must accept the licence and sign in | Loads a different model under the same name |
| You really want a random model (tests) | `allow_random_init=True` returns one, flagged `_pygeovision_is_random_init`, with a log line | Does it quietly |

## Choosing

| Situation | What to do |
|---|---|
| You have few labels | Fine-tune with a frozen encoder first (`freeze_backbone=True`), unfreeze later (`unfreeze_backbone_epoch=N`): [Training](../training/index.md) |
| You have no labels | Use CLIP zero-shot, embeddings with nearest-neighbour search, or a [pipeline](../pipelines/index.md) that needs no model |
| The model needs bands you do not have | DOFA accepts any band set; Prithvi needs the six HLS bands |
| You have radar | Prepare and map it first ([Radar](radar.md)); an optical model fed raw backscatter gives a confident, meaningless map |
| You want a vector per pixel and no GPU | [TESSERA or AlphaEarth](embeddings.md), then a simple classifier |
| You want object outlines to speed up labelling | [SAM](sam.md), with a human accepting each mask |
| Offline machine | Pre-download (`pygeovision models download NAME`, `load_prithvi_local`, `HF_HOME`) before disconnecting |
| Large downloads fill the disk | Weights live in `~/.cache/huggingface`; delete what you no longer use |

## What was verified, what was not

Everything that does not need a download was run here and is tested: band mapping and validation, shapes, heads, the saved-head files, the
multi-date tools, the embeddings wrappers (against stand-in modules), the radar mapping and adapters. **Not run here:** any real checkpoint (Prithvi,
DOFA, DINOv3, CLIP, SAM) and any accuracy figure, because the test environment has no network. Each page says which of its claims were checked and how.

## Related

- [Models by task](../models/tasks.md), [Choosing a model](../models/choosing.md), [Fine-tuning](../training/fine-tuning.md)
- [Vision-language models](../vision-language/index.md), [Classification with few labels](../classification/few-shot.md)
- [`pygeovision.models.foundation` in the Python API](../reference/api/builders-foundation.md)
