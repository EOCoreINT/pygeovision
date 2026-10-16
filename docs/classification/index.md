# Classification

Classification gives **one label to a whole image or chip**: "this tile is forest", "this scene is flooded", "this
chip contains a solar farm". It is the right tool when the answer is about the picture as a whole. When you need to know
*where* inside the picture, use [segmentation](../segmentation/index.md); when you need to count discrete objects, use
[detection](../detection/index.md).

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Models
:link: models
:link-type: doc
ResNet, DenseNet, EfficientNet, ConvNeXt, Swin, ViT, DINOv2 and CLIP.
:::

:::{grid-item-card} Train one
:link: train
:link-type: doc
A plain PyTorch loop with a PyGeoVision model, and how to check it.
:::

:::{grid-item-card} Predict
:link: predict
:link-type: doc
Zero-shot CLIP, a trained classifier and a grid of chips over a scene.
:::

:::{grid-item-card} Few-shot
:link: few-shot
:link-type: doc
Two to ten examples per class with a frozen DINOv2 and no training.
:::
::::

## What goes in and what comes out

Supervised classifiers (`resnet50`, `convnext-t`, `swin-t`, `vit-b16`, `dinov2-b`, …) take `(batch, bands, H, W)` and return
`(batch, classes)` scores:

| Model | Input | Output |
|---|---|---|
| `resnet50` (5 classes, 4 bands) | `(2, 4, 64, 64)` | `(2, 5)` |
| `vit-b16` (5 classes, 4 bands) | `(1, 4, 224, 224)` | `(1, 5)` |

ViT and Swin need the chip size they were built for: pass `img_size=` to `get_model` when yours differs.

```python
import torch
from pygeovision.models import get_model

model = get_model("resnet50", num_classes=5, in_channels=4, pretrained=False).eval()
probabilities = model(torch.rand(2, 4, 64, 64)).softmax(dim=1)     # (2, 5), each row sums to 1
```

## Four ways to classify

```{figure} ../_static/img/flow-classification.svg
:alt: Four routes to a classification: reference map, zero-shot, few-shot, trained classifier
:width: 100%
:class: pgv-figure

Four routes, from no labels at all to a fully trained classifier.
```

| You have | Use | Needs |
|---|---|---|
| Nothing, and a published map is enough | **Land cover from a reference map**: `pygeovision ai classify land-cover` | Network (downloads ESA WorldCover). Not a classification of *your* image |
| Nothing, and you want a first look | **Zero-shot CLIP**: `pygeovision ai classify scene --classes "..."` | CLIP weights (downloaded). Weakest on unusual sensors |
| 2 to 10 labelled chips per class | **[Few-shot](few-shot.md)** with frozen DINOv2 | The DINOv2 weights |
| Hundreds of labelled chips per class | **[Train a classifier](train.md)** | Labelled chips and a GPU helps |

## Zero-shot, with no training

CLIP compares an image with text descriptions.

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
(remote-sensing-tuned, see [CLIP and RemoteCLIP](../foundation-models/clip.md)).

## Land cover from a reference map, not a model

`pygeovision ai classify land-cover --input s2.tif --output lc.tif` downloads the ESA WorldCover map for the image's
extent (in Python, `client.classification.land_cover(path, year=2021)` chooses the year). It is a published product at 10 m,
appropriate for labels and context; it is not a classification of *your* image.

## Reading the result

Classifiers output **scores**, not certainty. A softmax probability of 0.9 means the network preferred that class by a
wide margin, not that it is right 90% of the time. For decisions:

- measure accuracy on a held-out set the model never saw (**overall accuracy**, and per-class **precision** and **recall**);
- look at the **confusion matrix** to see which classes are mixed up;
- calibrate or threshold on the probabilities if false alarms are costly.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| Every chip gets the same class | Too little training, a collapsed model, or very unbalanced classes | Check training accuracy per class; balance the classes or weight the loss; train longer |
| High training accuracy, poor validation accuracy | Over-fitting, or chips from the same scene in both sets | Split by scene or area; augment; use a smaller model or pretrained weights |
| `ValueError` about the window size (ViT/Swin) | Chip size differs from what the model was built for | `get_model(..., img_size=<your chip size>)` |
| Zero-shot results look random | The sensor or the bands are far from photographs, or the category words are ambiguous | Use RemoteCLIP, describe the categories in more detail, or train a classifier |
