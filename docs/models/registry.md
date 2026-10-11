# How the registries relate

PyGeoVision has more than one list of models. This is a consequence of how the package grew, and it matters in
practice: a name that works in one command may not exist in another. This page explains the lists, which command uses
which, how names correspond, and how to add a model of your own.

## Three lists, three jobs

| List | Names look like | Entries | Job | Browse with |
|---|---|---:|---|---|
| **Native registry** (`pygeovision.models`) | `unet-r50`, `segformer-b2` (hyphens) | 79 | Build an architecture: `get_model(name, ...)` | `pygeovision models …` |
| **Legacy engine registry** (`pygeovision.ai.models`) | `unet_resnet50`, `segformer_b2` (underscores) | 14 | The default home of the names pipelines and `infer` accept | `pygeovision ai-models …` |
| **Model zoo** (`pygeovision.ai.models.zoo`) | `sam_vit_h`, `prithvi_seg` | 96 | Describe models: size, tags, uses. **Does not build anything** | `pygeovision zoo …` |

The [catalogue](catalog.md) lists every entry of all three.

```{admonition} Which one should I use?
:class: tip
Use the **native registry** for anything you build yourself in Python (`get_model`). For the **commands**, use the
name the command's help text shows. Both spellings work in commands that load your checkpoint, because those commands
look in the legacy registry first and then in the native one (see below).
```

## Which command uses which

| You run | Model name is looked up in |
|---|---|
| `pygeovision models list / info / summary / download` | Native registry |
| `pygeovision ai-models list / info / cache` | Legacy registry |
| `pygeovision zoo list / info / search / top / stats` | Zoo |
| `pygeovision infer predict`, `infer batch` | Legacy first, then native (`ModelHub.load`) |
| `pygeovision ai infer`, `ai segment custom`, `ai detect custom` | Legacy first, then native |
| `pygeovision channel … --checkpoint`, YAML `infer` steps | Legacy first, then native |
| `pygeovision edge export-onnx` | Native only |
| `client.segmentation.custom(...)`, `client.detection.custom(...)` | Legacy first, then native |
| `GeoTrainer(model, ...)` | None: you pass a model **object** |
| Point-cloud segmentation (`client.pointcloud`) | Native |

The zoo is never used to build anything.

### What "legacy first, then native" means

`ModelHub().load("name", checkpoint=..., num_classes=..., in_channels=...)` tries the legacy registry. If the name is
not there, or building fails, it tries the native registry with the same name. It does **not** silently choose a
different architecture: a failure raises `ModelLoadError` naming every attempt. (An `allow_fallback=True` option exists
for scripts that accept a same-task substitute; it is off by default and flags the result.)

## Names that mean the same architecture

If you train with a native name and later run with a command, the command must build the **same network**. These pairs
produce identical layer names and shapes (verified by building both with 4 bands and 3 classes), so a checkpoint from
one loads into the other:

| Legacy name (commands, pipelines) | Native name (`get_model`) | Interchangeable? |
|---|---|---|
| `unet_resnet50` | `unet-r50` | ✅ |
| `unet_efficientnet_b4` | `unet-efficientb4` | ✅ |
| `segformer_b2` | `segformer-b2` | ✅ |
| `segformer_b5` | `segformer-b5` | ✅ |
| `resnet50_cls` | `resnet50` | ✅ |
| `fcos_resnet50` | `fcos-r50` | ✅ |
| `deeplabv3plus_resnet101` | `deeplab-r101` | ❌ DeepLabV3+ is not DeepLabV3 |
| `vit_b16_cls` | `vit-b16` | ❌ Different ViT implementations |
| `changeformer` | `changeformer-mit-b0` | ❌ Different networks, neither is the published ChangeFormer |

The simplest rule: **train and run with the same name**, and keep `--num-classes` and `--in-channels` the same. A
mismatch fails with `size mismatch` or `Missing key(s)` when the checkpoint is loaded. That error is the safeguard
working: the alternative would be a model that loads partly and predicts nonsense.

## `get_model`

```python
from pygeovision.models import get_model

model = get_model(
    "segformer-b2",
    num_classes=7,          # output classes (include background if you want it predicted)
    in_channels=6,          # bands in
    pretrained=True,        # download published weights; error if that fails
    device="cuda",          # optional: move to this device
)
```

| Argument | Default | Meaning |
|---|---|---|
| `name` | required | A native registry name |
| `num_classes` | `2` | Output classes. For detectors, the number of object classes (background is added where the architecture needs it) |
| `in_channels` | `4` | Input bands. If not 3, the first layer is widened; extra bands start from the mean RGB filter and are **not pretrained** |
| `pretrained` | `True` | Load published weights. Failure raises `ModelLoadError`; random weights are never substituted silently |
| `device` | `None` | Move the model to this device |
| `allow_random_init` | `False` | Opt in: if pretrained weights cannot be loaded, return an architecture-only model flagged `_pygeovision_is_random_init` |
| `img_size` | model default | For window- and position-embedding models (Swin, ViT, DINOv2): the chip size you will use |

### What `get_model` raises

| Exception | Meaning | What to do |
|---|---|---|
| `KeyError` | Unknown name | `pygeovision models list`; check hyphens vs underscores |
| `NotImplementedError` | Listed for discovery; no build path | Read the message: it names a working alternative |
| `ModelLoadError` | Weights or Hub config could not be fetched | Connect, set `HF_HOME` (or `HF_HUB_CACHE`) to a populated cache, or use `pretrained=False` |
| `ValueError` (Swin/ViT size) | Fixed-resolution model, chip size differs | Pass `img_size=<your chip size>` |
| `ValueError` (Prithvi) | No architecture-only Prithvi | Use `pretrained=True` or `allow_random_init=True` |
| `ImportError` | An optional package is missing (`timm`, `torchgeo`, `open_clip_torch`) | Install the package named in the message |

## Weights: where they are stored

| What | Where | Controlled by |
|---|---|---|
| Hugging Face weights (SegFormer, ViT, DINOv2, DETR, SAM, Prithvi, …) | The Hugging Face cache, `~/.cache/huggingface` | `HF_HOME` |
| `timm` weights (ResNet, Swin, ConvNeXt, EfficientNet) | The Hugging Face cache too (current `timm` fetches from the Hub) | `HF_HOME` |
| torchvision weights (Faster R-CNN, FCOS, FCN) | The torch cache, `~/.cache/torch` | `TORCH_HOME` |
| Checkpoints you train | Your training `output_dir/checkpoints/` | `TrainingConfig(output_dir=…)` |
| Checkpoints saved with `ModelHub.save` | `~/.pygeovision/models/` | `ModelHub(cache_dir=…)` |

`pygeovision models download NAME` pre-fetches a Hugging Face model into the Hugging Face cache, so you can go offline
afterwards; with `--cache-dir` it saves to the folder you name instead (then set `HF_HUB_CACHE` to that folder to use it).
`timm` models are fetched automatically on first use.

## Checkpoint format

`GeoTrainer` writes checkpoints as a dictionary:

```python
{"epoch": 12, "model_state_dict": {...}, "metrics": {...}, "metric": "val_iou", "score": 0.81}
```

Commands and `ModelHub.load` accept that dictionary, a plain state dict, or a dictionary whose weights sit under one
of the keys `model`, `model_state_dict`, `state_dict` or `network`. They load with `strict=True`.

`last.pth` additionally holds the optimizer, scheduler and history, for resuming training. It loads in commands too, but
`best.pth` is the file to deploy.

## Adding your own model

Any `torch.nn.Module` works with `GeoTrainer` and `TiledInference` as it is. Registering it makes it available by
name to `ModelHub.load`, so you do not have to rebuild it by hand each time.

```python
import torch.nn as nn
from pygeovision.ai.models.registry import registry, ModelInfo

class TinyNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, num_classes, 1))
    def forward(self, x):
        return self.body(x)

registry.register(ModelInfo(
    name="tiny_net", task="segmentation", architecture="TinyNet", input_bands=4,
    # The factory receives num_classes, in_channels and pretrained, so accept **kwargs.
    factory_fn=lambda in_channels=4, num_classes=2, **kw: TinyNet(in_channels, num_classes),
))
```

Then:

```python
from pygeovision.ai.models.hub import ModelHub

model = ModelHub().load("tiny_net", num_classes=2, in_channels=4)                # untrained
model = ModelHub().load("tiny_net", checkpoint="tiny.pth", num_classes=2, in_channels=4)
```

Expected behaviour (this is the tested flow): the first call returns a `TinyNet`; with a checkpoint, the weights load
strictly, and asking for a different `num_classes` than the checkpoint was trained with raises

```text
ModelLoadError: checkpoint tiny.pth does not match 'tiny_net' (3 classes, 4 bands): Error(s) in loading state_dict
for TinyNet: size mismatch for body.2.weight: ...
```

```{admonition} Registration lives for one Python process
:class: note
`registry.register(...)` changes the registry of the running program. A separate `pygeovision infer predict …` command
starts a new process and does not know `tiny_net`. For your own models, run inference from Python
(`TiledInference(model=model, …)`), or put the registration in a module you import in your own entry script.
```

Registering a name that already exists replaces it and logs a warning.

## The zoo

`pygeovision zoo list --task segmentation --tag transformer --max-params 100` filters descriptions; `zoo info NAME`
prints the full record; `zoo top TASK` ranks entries that have published weights by a simple score in which a
Hugging Face checkpoint and a larger parameter count both raise the rank. That ranking is a **starting point for reading, not a benchmark**: it knows nothing about your data. Zoo
names cannot be passed to `--model`.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `Model 'unet-r50' not found in registry` from `ai-models info` | `ai-models` shows only the legacy list | Use `models info unet-r50`, or the legacy name `unet_resnet50` |
| `Could not build 'X' (…)` from a command | Name in neither registry, or the build failed | `pygeovision models list`; read each attempt listed in the message |
| `size mismatch for …` loading a checkpoint | Different architecture, classes or bands than in training | Use the training model name, `--num-classes` and `--in-channels` |
| A checkpoint trained as `deeplab-r101` fails as `deeplabv3plus_resnet101` | They are different networks | Use the exact name you trained |
| `zoo` shows a model but `--model` rejects it | The zoo only describes models | Pick a name from the native or legacy list |
| Registered model not found in the CLI | Registration is per process | Run from Python, see above |
