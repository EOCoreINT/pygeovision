# Unavailable models

The registry names a few models so that you can **find** them and see why they are not offered. Building any of them raises
`NotImplementedError` with the reason. Nothing is substituted: asking for `changeformer-mit-b4` never quietly gives you a
different network under that name, because a model with the wrong architecture produces plausible-looking maps that are wrong.
The [model catalogue](catalog.md) marks these entries *listed only*, and each task's models page shows them in its table.

```python
from pygeovision.models import get_model

get_model("changestar-r18")
# NotImplementedError: 'changestar-r18': ChangeStar has no implementation in torchvision/timm/transformers and none in this
# codebase (use ChangeFormer, BIT or DSAMNet). Provide your own model rather than silently getting a different architecture.
```

## The nine entries

| Name | Task | What it is | Why it cannot be built | Use instead |
|---|---|---|---|---|
| `rf-detr-b`, `rf-detr-l` | Detection | Roboflow's real-time DETR (29 M and 128 M parameters) | It is distributed in Roboflow's own package; no builder, `hf_id` or `timm_id` is registered here | [`rt-detr-l`](../detection/models.md#detr-and-rt-detr) for an end-to-end detector, or `faster-rcnn-r50` |
| `dinov3_segmentor` | Segmentation | A DINOv3 backbone with a segmentation head | The head needs a DINOv3 backbone, which lives in the foundation package, not in the segmentation registry | Use [`DINOv3Backbone`](../foundation-models/dinov3.md) features with a head of your own, or `segformer-b2` |
| `changestar-r18` | Change detection | ChangeStar, single-temporal-supervised change detection | No implementation exists in torchvision, timm or transformers, and none in this package | `bit-r50`, `dsamnet`, `changeformer-mit-b0` ([Change-detection models](../change-detection/models.md)) |
| `changeformer-mit-b4` | Change detection | The published ChangeFormer with a MiT-B4 backbone (67.4 M) | Only the compact B0 network is implemented; building it under the B4 name would be mislabelling | `changeformer-mit-b0`, `bit-r50` or `dsamnet` |
| `prithvi_burn_scar` | Segmentation | A burn-scar fine-tune of Prithvi, distributed for TerraTorch / MMSegmentation | This package cannot load that format | Fine-tune [`prithvi-100m`](../foundation-models/prithvi.md) on your own burn labels, or use the [`wildfire_severity` pipeline](../pipelines/catalog/wildfire_severity.md) (dNBR, no weights) |
| `satlas-pretrain` | Foundation | SatlasPretrain, a large remote-sensing pretraining set and models | No verified public checkpoint or construction code here | [Prithvi-EO](../foundation-models/prithvi.md) or [DOFA](../foundation-models/dofa.md) |
| `chatearthnet` | Vision-language | A remote-sensing image-text dataset and model | No verified public checkpoint or construction code here | [Moondream](../vision-language/captions.md) for captions, [CLIP](../vision-language/clip.md) for search |
| `lisat-7b` | Vision-language | LISAt-7B, a 7-billion-parameter geospatial reasoning model that outputs segmentation masks (Quenum et al., 2025) | It is a custom LISA-style multimodal LLM with a SAM decoder, not a standard `transformers` class. Loading needs the authors' `lisat_code` repository, `flash-attn` and a separate RemoteCLIP checkpoint | [SAM](../foundation-models/sam.md) with [CLIP](../vision-language/clip.md) for text-guided outlines |

`lisat-7b` is the one that is *real and published* but too heavy to wrap soundly; its weights are at `jquenum/LISAt-7b`. If you
need it, follow the authors' instructions and treat it as an external tool.

## Why list a model at all if it cannot be built?

1. **So the search is honest.** Typing `pygeovision models list --task change_detection` shows every name people expect to see, with
   a status, rather than leaving you to wonder whether the package has "ChangeFormer B4" and forgot to mention it.
2. **So the answer is one command away.** The error text says what to use instead.
3. **So adding one later is a small change.** A builder is registered, the status flips to *builds*, and the page updates itself,
   because [the per-task model pages are generated](catalog.md) from the registry.

## How to bring your own

When the model you need is not here, `GeoTrainer` and `TiledInference` accept **any** PyTorch `nn.Module` with the right input and
output shapes ([the shapes for each task](tasks.md)). You do not have to register it:

```python
import torch
from pygeovision.training import GeoTrainer, TrainingConfig

my_model = MyNetwork(...)                          # your code: (B, bands, H, W) -> (B, classes, H, W)
trainer = GeoTrainer(my_model, TrainingConfig(num_classes=3, in_channels=4, output_dir="runs/mine"))
trainer.fit(train_ds, val_ds)
```

To make a model available by name everywhere (CLI, pipelines), register a builder; see [Adding your own model](registry.md#adding-your-own-model).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: No build path is registered for ...` | A listed-only entry | Read the message; use the alternative in the table, or provide your own model |
| A model appears in `models list` but `models info` shows no weights | Listed only | Same |
| You need one of these models for a project | It is a real model but not wrapped here | Use the authors' code as an external step, and bring the result in as files (rasters or vectors) |

## Related

- [Model catalogue](catalog.md) and [Models by task](tasks.md).
- [How the registries relate](registry.md).
