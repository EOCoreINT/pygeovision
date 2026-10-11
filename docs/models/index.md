# Models

A *model* in PyGeoVision is a neural-network architecture you can build, train, fine-tune and run on satellite imagery.
This section explains what the package gives you, what it deliberately does **not** give you, and how to choose and use
a model from first command to saved checkpoint.

```{toctree}
:hidden:
:maxdepth: 1

registry
choosing
tasks
foundation
catalog
```

| Page | Read it when |
|---|---|
| [How the registries relate](registry.md) | A model name works in one command and not in another, or you want to add your own model |
| [Choosing a model](choosing.md) | You know your task and data and need to pick an architecture |
| [Models by task](tasks.md) | You want the input and output of each kind of model, with a working example |
| [Foundation models](foundation.md) | You want Prithvi, DOFA, DINOv3, CLIP, SAM or satellite embeddings, or to use radar with an optical foundation model |
| [Model catalogue](catalog.md) | You want the full list: every name, size, weight source and whether it builds |

## The most important thing to know

**PyGeoVision ships architectures, not trained task models.**

An architecture such as `unet-r50` is a recipe for a network. It becomes useful for a task such as "map crop types in
Ghana" only after it has been *trained* on labelled examples of that task. The package therefore:

- gives you the architectures and the tools to train them ([Training](../training/index.md));
- downloads **pretrained starting points** where a real public checkpoint exists (ImageNet weights for ResNet,
  Hugging Face weights for SegFormer, HLS weights for Prithvi). These know how to see edges and textures; they do not
  know your classes;
- refuses to run an untrained network and call the output a result. Commands that need your model stop with a message
  that tells you what to supply.

This is why seven of the [40 pipelines](../pipelines/index.md) need `--checkpoint`, and why the other 33 use
published methods (indices, thresholds, reference maps) that need no training at all. If you only want a flood map or a
burn-severity map, you do not need this section.

```{admonition} A model that "runs" is not a model that "works"
:class: warning
An untrained or wrongly trained network produces a raster of plausible-looking classes. Nothing about the file tells
you it is wrong. That is why PyGeoVision never substitutes a model, never falls back to random weights without telling
you, and loads your checkpoint **strictly**: if its shapes do not match the architecture, class count and band count,
loading fails.
```

## Where models come from

```text
   public pretrained weights            your labelled chips
   (ImageNet, ADE20K, HLS, ...)          (GeoTIFF image + label pairs)
              \                                /
               v                              v
        get_model(name, pretrained=True)   GeoTrainer.fit(...)   --->  best.pth (your checkpoint)
                                                                           |
                  pygeovision infer predict / channel ... --checkpoint <---+
```

1. **Pick an architecture** from the [catalogue](catalog.md) with `pygeovision models list`.
2. **Train it** on your labelled data (or fine-tune a pretrained one). You get a `.pth` file.
3. **Run it** with `pygeovision infer predict`, a pipeline's `--checkpoint`, or Python ([Inference](../inference/index.md)).

If you have no labels yet, [Labeling](../labeling/index.md) shows how to make training data from OpenStreetMap, ESA
WorldCover, Dynamic World and building-footprint datasets.

## Five-minute tour

List what is available, look at one model, build it:

```bash
pygeovision models summary                  # counts per task
pygeovision models list --task segmentation --max-params 40
pygeovision models info segformer-b2        # size, weight source, notes
```

```python
from pygeovision.models import get_model

# 4 bands (for example blue, green, red, near-infrared), 3 output classes.
model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
print(type(model).__name__)                 # Unet
```

Everything that builds returns a standard `torch.nn.Module`. A segmentation model maps a tensor of shape
`(batch, bands, H, W)` to class scores of shape `(batch, classes, H, W)`:

```python
import torch
scores = model.eval()(torch.rand(2, 4, 64, 64))
print(scores.shape)                         # torch.Size([2, 3, 64, 64])
```

Argument meanings:

| Argument | Meaning |
|---|---|
| `num_classes` | How many classes the network predicts. **Count the background** if you want it predicted (a map with *water* and *not water* has 2 classes) |
| `in_channels` | How many bands go in. A pretrained network was trained on 3 bands, so for other counts the first layer is widened and the *extra bands start untrained* |
| `pretrained` | `True` downloads published weights; if that fails you get an error, never random weights. `False` builds the architecture only |

## What to do when…

| Situation | What to do |
|---|---|
| You do not know which model to use | [Choosing a model](choosing.md) gives a decision table by task, data volume and hardware |
| `Model '…' not found` | Names differ between registries (`unet-r50` and `unet_resnet50`). [How the registries relate](registry.md) shows which command takes which |
| `NotImplementedError` from `get_model` | The name is listed for discovery only; the message names a working alternative. The [catalogue](catalog.md) marks these as *listed only* |
| `ModelLoadError … could not load pretrained weights` | No network, or the Hub repository is gated. Use `pretrained=False` and train, or pre-download the weights |
| A pretrained download is slow or blocked | Set `HF_HOME` to a folder that already contains the weights, or build with `pretrained=False` |
| Prediction looks random | The checkpoint is not trained for this data, or the input is scaled differently from training. See [Inference](../inference/index.md#matching-the-input-to-training) |
| You need a model that is not here | Register your own `nn.Module`: [How the registries relate](registry.md#adding-your-own-model) |
