# Choosing a model

There is no best model; there is the best model *for your task, your data and your hardware*. This page turns those
three into a decision, and says plainly when the right answer is **not to train a model at all**.

```{admonition} Advice here is a starting point, not a benchmark
:class: note
PyGeoVision does not publish accuracy numbers for architectures on your data, because they depend on it. The
recommendations below are common practice for each situation. Compare two or three candidates on a held-out
validation area before committing ([Validation](../data/validation.md), [Benchmark](../experiments/benchmark.md)).
```

## Step 1: do you need a model?

Many questions are answered without training, by methods that need no labels.

| You want | Try first | Needs training? |
|---|---|---|
| Water, flood extent | [`water_bodies`](../pipelines/catalog/water_bodies.md), [`flood_mapping`](../pipelines/catalog/flood_mapping.md), [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md) | No |
| Burn severity, vegetation health, snow | [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md), [`vegetation_indices`](../pipelines/catalog/vegetation_indices.md), [`snow_cover`](../pipelines/catalog/snow_cover.md) | No |
| General land cover | [`land_cover`](../pipelines/catalog/land_cover.md) (published reference maps) | No |
| Building footprints | [`building_footprints`](../pipelines/catalog/building_footprints.md) (Microsoft, Google, OSM datasets) | No |
| Change between two dates | `pygeovision ai change` with the default `spectral_diff` method | No |
| Classify one image by words ("forest", "port") | CLIP zero-shot: [`ai classify scene`](../classification/predict.md) | No |
| Find similar places | [Embeddings](../foundation-models/embeddings.md) | No |
| **Your own classes**, local crops, a feature no public dataset maps | A trained model | **Yes** |

If a no-training method answers the question well enough, use it: it is faster, reproducible and has no hidden failure
mode. Reach for a model when you need classes or accuracy the reference products cannot give.

## Step 2: pick the task type

| Your output is | Task | Typical models |
|---|---|---|
| A class for **every pixel** (land cover, crop type, water, burned) | Segmentation | `unet-r50`, `segformer-b2`, `deeplab-r50` |
| One label **per image or chip** ("this chip has a mine") | Classification | `resnet50`, `convnext-t`, `swin-t` |
| **Boxes around objects** (vehicles, ships, solar panels) | Detection | `faster-rcnn-r50`, `fcos-r50`, `rt-detr-l` |
| What **changed** between two images | Change detection | `bit-r50`, `dsamnet`, or two-date segmentation |
| A class per **3-D point** (LiDAR) | Point cloud | `pointnet2-ssg`, `randlanet` |
| A **sharper** image | Super-resolution | `swinir-m` (RGB only) |
| **Text** about an image | Vision-language | `moondream2`, CLIP family |

Detection is the hardest to get right. If your objects are large enough to appear as blobs in a mask (buildings,
ponds), segmentation followed by [vectorising](../data/postprocessing.md) often works better and needs simpler labels.

## Step 3: match the architecture to your data

### How much labelled data do you have?

| Labelled chips (image + mask pairs) | Do this |
|---|---|
| Fewer than ~100 | Do not train from scratch. Fine-tune a **pretrained** model, freeze the encoder at first (`freeze_backbone=True`), or use a [foundation model](../foundation-models/index.md) with a small head |
| ~100 to ~2,000 | Fine-tune a pretrained model. `segformer-b0/b2` or `unet-r50` with pretrained encoder, augmentation on |
| More than ~2,000 | Any of the above; larger models (`segformer-b5`, `swin-b`) start to pay off |

These counts are rules of thumb. What matters more is that the chips cover the variety you will meet when you run the
model (seasons, regions, sensors). A model trained on one valley in June will not generalise to another country in
December, however many chips it saw.

### How many bands?

| Your input | Choice |
|---|---|
| RGB, or RGB + NIR (3–4 bands) | Any model. Pretrained weights fit RGB best |
| 6–13 multispectral bands | Fine, but the extra bands start untrained. Expect to fine-tune longer. [Prithvi](../foundation-models/index.md) and [DOFA](../foundation-models/index.md) are designed for multispectral input |
| Radar (VV, VH) | Train on radar directly, or use the [SAR adapters](../foundation-models/radar.md#radar-with-optical-foundation-models). Do not feed raw dB values to an optical network |
| Very many bands (hyperspectral) | Segmentation models still work (`in_channels=…`); consider reducing bands first |

### What hardware do you have?

Memory use grows with model size, chip size and batch size. As orientation (training, 256 × 256 chips):

| Hardware | Sensible models | Notes |
|---|---|---|
| CPU only | `segformer-b0` (3.8 M), small chips | Works for experiments and small data; slow |
| 8 GB GPU | `unet-r50`, `segformer-b2`, `deeplab-r50` at batch size 4–8 | Lower the batch size before the chip size |
| 16–24 GB GPU | Any single model; `segformer-b5`, `swin-b`, `upernet-swin-b` | Use mixed precision (`precision="fp16"`) |
| Several GPUs | Distributed training (`strategy="ddp"`) | [Training: advanced](../training/fine-tuning.md) |

Inference needs much less than training and can run on a CPU if you accept the time ([Inference](../inference/index.md)).

## Step 4: a short list per task

### Segmentation

| Model | Params (M) | Good when | Watch out |
|---|---:|---|---|
| `segformer-b0` | 3.8 | Small data, CPU, quick experiments | Lower accuracy ceiling |
| `segformer-b2` | 27.5 | A strong default with pretrained weights | Needs the Hugging Face download |
| `unet-r50` | 32.5 | Sharp object boundaries (buildings, roads); the classic choice | Heavier on memory than SegFormer-B2 at equal accuracy |
| `unet-efficientb4` | 20.2 | Similar to U-Net/ResNet-50, lighter | |
| `deeplab-r50` | 39.6 | Large, smooth regions (land cover) | Coarser boundaries |
| `upernet-swin-b` | 120 | Large data, big GPU | Heavy |
| `fcn-r50` | 32.9 | A simple baseline to compare against | Lowest ceiling of the set |

Start with `segformer-b2` or `unet-r50` unless you have a reason not to.

### Classification

`resnet50` is the safe baseline; `convnext-t` and `swin-t` are modern equivalents; `dinov2-b` features with a small
head work well with very few labels ([Foundation models](../foundation-models/index.md)). Classifiers expect a fixed chip size; Swin and
ViT need `img_size=` set to your chip size.

### Detection

`faster-rcnn-r50` and `fcos-r50` (torchvision) are the dependable choices and train with `GeoTrainer(task="detection")`.
`rt-detr-l` and `detr-*` come from the Hugging Face stack; DETR variants train slowly and need more data. Detection checkpoints cannot be run with `pygeovision infer predict`, which is for models that output class
scores per pixel: see [Inference](../detection/predict.md).

### Change detection

Try `ai change` with `spectral_diff` first. If you need *semantic* change ("forest became farmland" rather than "pixels
differ"), train `bit-r50` or `dsamnet`, or segment each date with the same model and compare the maps. (The
[`landcover_change`](../pipelines/catalog/landcover_change.md) pipeline compares published annual land-cover maps rather
than running a model.)

```{admonition} The ChangeFormer names
:class: warning
`changeformer-mit-b0` is a small (about 0.8 M parameter) siamese network in the spirit of ChangeFormer, trained from
scratch. It is **not** the published transformer ChangeFormer, and `changeformer-mit-b4` is listed only. The legacy
`changeformer` is a different network again (two SegFormers fused by a convolution). If you need a well-established
architecture, prefer `bit-r50` or `dsamnet`.
```

## Step 5: check your choice

Before spending hours on training:

1. **Build it and push a batch through** with your band and class counts; this catches shape problems in seconds.

   ```python
   import torch
   from pygeovision.models import get_model

   model = get_model("segformer-b2", num_classes=5, in_channels=6, pretrained=False).eval()
   print(model(torch.rand(2, 6, 256, 256)).shape)       # torch.Size([2, 5, 256, 256])
   ```

2. **Overfit a handful of chips.** Train on 10 chips for 30 epochs. If the loss does not fall towards zero, something
   is wrong with the data or labels, not with the architecture ([Training](../training/index.md#checking-that-training-works)).
3. **Look at predictions on validation chips**, not only the metric.

## What to do when…

| Situation | What to do |
|---|---|
| Training loss falls but validation IoU stays low | Too little data or too different between train and validation areas. Add augmentation, more chips, or a smaller model |
| Out of GPU memory | Smaller `batch_size`, then smaller chips, then `precision="fp16"`, then a smaller model |
| Predictions are all one class | Class imbalance. Use a [loss that handles it](../training/losses.md) (Dice, Focal, class weights) |
| Boundaries are blurry | A U-Net-type model, or a boundary-aware loss |
| A model you want is *listed only* | Use the alternative the error message names, or [bring your own](registry.md#adding-your-own-model) |
