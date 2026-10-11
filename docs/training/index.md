# Training a model

This section is for the moment the published methods are not enough: you have your own classes, your own imagery, and
you want a network that predicts them. It takes you from a folder of labelled images to a checkpoint file that
`pygeovision infer predict`, a pipeline's `--checkpoint`, or your own Python code can run.

```{toctree}
:hidden:
:maxdepth: 1

data
config
losses
advanced
```

| Page | Read it when |
|---|---|
| This page | You want the whole workflow, one working example, and how to tell whether training is going well |
| [Preparing training data](data.md) | You need to build the image and label chips, split them, and scale them correctly |
| [The training configuration](config.md) | You want every `TrainingConfig` setting explained, with its default and what it really does |
| [Loss functions and class imbalance](losses.md) | Your classes are rare, or the model predicts one class everywhere |
| [Advanced training](advanced.md) | Mixed precision, resuming, staged fine-tuning, several GPUs, hyper-parameter search, experiment tracking, exporting |

## Why train at all

Reach for training only when [a method that needs no training](../models/choosing.md#step-1-do-you-need-a-model) will
not do. Training costs labels (the expensive part), time, and care: a network that has not been checked produces
maps that *look* right and are not. The reward is a model for **your** classes, **your** sensor and **your** region.

## The workflow in one picture

```text
 1 labels + imagery        2 chips                3 model              4 train             5 check            6 use
 (GeoTIFF pairs,    -->  (image/label   -->   get_model(...)   -->  GeoTrainer     -->  read the log,  -->  infer predict,
  same grid)              tiles, split)        pretrained or        .fit(train, val)     look at maps       pipelines,
                                               from scratch                                                    Python
```

1. **Get labelled data**: [Labeling](../labeling/index.md) shows how to make labels from OpenStreetMap, WorldCover,
   building-footprint datasets or by hand. Labels and imagery must be on the same pixel grid.
2. **Cut chips and split them**: [Preparing training data](data.md).
3. **Pick a model**: [Choosing a model](../models/choosing.md).
4. **Train** with `GeoTrainer` (this page).
5. **Check** the result before trusting it (below).
6. **Run it**: [Inference](../inference/index.md).

```{admonition} The command "pygeovision ai train" does not train
:class: warning
`pygeovision ai train segmentation --data … --output model.pth` only prints a ready-to-edit configuration and the
Python lines to run. It writes no model. Training is done from Python with `GeoTrainer`, as below, because it needs
decisions about data, splits and losses that a one-line command cannot hide.
```

## A complete, working example

Install the training extras first: `pip install "pygeovision[train]"` (PyTorch, `segmentation-models-pytorch`,
`albumentations`, `optuna`, `mlflow`, `wandb`). The example assumes this folder layout (see
[Preparing training data](data.md) for how to build it):

```text
data/
├── train/
│   ├── images/   s0.tif  s1.tif  s2.tif      4-band float32 GeoTIFFs, values 0-1
│   └── labels/   s0.tif  s1.tif  s2.tif      1-band uint8 GeoTIFFs, 0 = background, 1 = class
└── val/
    ├── images/   s0.tif
    └── labels/   s0.tif
```

```python
import logging
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig
from pygeovision.training.data import GeoSegDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")   # so you can see the training log

train = GeoSegDataset("data/train/images", "data/train/labels", chip_size=128)
val   = GeoSegDataset("data/val/images",   "data/val/labels",   chip_size=128)
print("chips:", len(train), len(val))

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)

cfg = TrainingConfig(
    output_dir="runs/demo",        # everything is written under this folder
    task="segmentation",
    num_classes=2,                 # background counts
    in_channels=4,
    max_epochs=10,
    batch_size=2,
    learning_rate=3e-4,
    warmup_epochs=0,
    num_workers=0, pin_memory=False, persistent_workers=False,   # simplest settings for a laptop
    devices="cpu",
)
result = GeoTrainer(model, cfg).fit(train, val)
print(result["best_checkpoint"], result["best_metric_name"], result["best_metric_value"])
```

```{admonition} About the numbers on this page
:class: note
The output below comes from a real run on **synthetic** data (three 256 × 256 four-band scenes with a rectangle
as the target), on a CPU, with `pretrained=False`. It shows what the log *looks like* and how to read it. It says
nothing about how well PyGeoVision will do on your imagery. With real data use `pretrained=True`, more chips and a
GPU.
```

Expected output (abridged; the number of chips and the exact values will differ on your data):

```text
chips: 12 4
Starting training: 10 epochs | device=cpu | precision=auto | strategy=auto
Epoch 1/10 | loss=0.4674 | val_loss=0.5063 | val_iou=0.4127 | lr=3.00e-04
Epoch 2/10 | loss=0.2537 | val_loss=0.4740 | val_iou=0.4154 | lr=2.93e-04
Epoch 3/10 | loss=0.1757 | val_loss=0.4906 | val_iou=0.4102 | lr=2.71e-04
Epoch 4/10 | loss=0.1385 | val_loss=0.4529 | val_iou=0.4101 | lr=2.38e-04
Epoch 5/10 | loss=0.1047 | val_loss=0.2827 | val_iou=0.4318 | lr=1.97e-04
Epoch 6/10 | loss=0.0927 | val_loss=0.1437 | val_iou=0.8770 | lr=1.50e-04
Epoch 7/10 | loss=0.0795 | val_loss=0.1046 | val_iou=0.9637 | lr=1.04e-04
Epoch 8/10 | loss=0.0718 | val_loss=0.0921 | val_iou=0.9791 | lr=6.26e-05
Epoch 9/10 | loss=0.0712 | val_loss=0.0848 | val_iou=0.9745 | lr=2.96e-05
Epoch 10/10 | loss=0.0694 | val_loss=0.0881 | val_iou=0.9665 | lr=8.32e-06
Training complete: 1.1 min | best=val_iou=0.9791
runs/demo/checkpoints/epoch_0008_val_iou_0.9791.pth val_iou 0.9791431427001953
```

### What each part of the example does

| Line | Why it is there |
|---|---|
| `GeoSegDataset(images, labels, chip_size=128)` | Cuts every scene into 128 × 128 tiles, pairs each with its label tile, and hands them to PyTorch one at a time. Details in [Preparing training data](data.md#geosegdataset) |
| separate `train` and `val` datasets | `val` is never learned from. It is the only honest measure of how the model does on imagery it has not seen |
| `get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)` | Builds the network. `num_classes` and `in_channels` **must** match your labels and bands. `pretrained=True` starts from ImageNet weights, which is almost always better with real data |
| `TrainingConfig(...)` | One object holding every setting. The full list is in [The training configuration](config.md) |
| `GeoTrainer(model, cfg).fit(train, val)` | The loop: forward, loss, backward, optimiser step, validation, checkpoint, early stopping |

### What you get on disk

```text
runs/demo/
└── checkpoints/
    ├── best.pth                         a copy of the best epoch: this is the file to deploy
    ├── epoch_0008_val_iou_0.9791.pth    the best few epochs, named by score (save_top_k, default 3)
    ├── epoch_0009_val_iou_0.9745.pth
    ├── epoch_0010_val_iou_0.9665.pth
    └── last.pth                         the newest state with optimiser and history: used to resume
```

`best.pth` and the `epoch_*` files hold the model weights, the epoch and the metrics. `last.pth` is about three times
larger because it also carries the optimiser, scheduler and history, so a stopped run can continue
([Advanced training](advanced.md#resuming-an-interrupted-run)).

### What `fit` returns

A dictionary:

| Key | Meaning |
|---|---|
| `epochs_trained` | Epochs actually run (fewer than `max_epochs` if early stopping triggered) |
| `duration_seconds` | Wall-clock time |
| `best_checkpoint` | Path of the best epoch file (`best.pth` is a copy of it) |
| `best_metric_name`, `best_metric_value` | The metric used to choose it and its value |
| `best_val_iou` | The same number under its older name (`best_val_map50` for detection) |
| `last_checkpoint` | Path of `last.pth` |
| `resumed_from_epoch` | `None`, or the epoch a resumed run continued from |
| `history` | Dictionary of lists, one value per epoch: `train_loss`, `val_loss`, `val_iou`, `val_map50` |

### Using the result

```bash
pygeovision infer predict scene.tif -m unet-r50 --checkpoint runs/demo/checkpoints/best.pth \
        -c 2 --in-channels 4 --chip-size 128 --overlap 16 --no-normalise -o prediction.tif
```

Three things must agree between training and prediction, or loading fails or the map is wrong: the **model name**,
the **number of classes** and **bands**, and **how the pixel values were scaled**. The last is the one people miss;
see [Inference: matching the input to training](../inference/index.md#matching-the-input-to-training). In the example
above `--no-normalise` is needed because the training chips were used as stored.

## Reading the training log

Each epoch prints one line:

```text
Epoch 6/10 | loss=0.0927 | val_loss=0.1437 | val_iou=0.8770 | lr=1.50e-04
```

| Field | Meaning | Healthy behaviour |
|---|---|---|
| `loss` | Average training loss over the epoch | Falls steadily, quickly at first, then slowly |
| `val_loss` | Loss on the validation chips | Falls with `loss`, staying somewhat above it |
| `val_iou` | Mean intersection-over-union over classes on validation: 1.0 is perfect, 0 is none. (Detection prints `val_map50` instead) | Rises, then flattens |
| `lr` | Learning rate now | Follows the schedule (here a cosine decay towards `min_lr`) |

Do not judge a run by `loss` alone: it can reach zero by memorising the training chips. **`val_iou` (or `val_map50`)
on an area the model never trained on is the number that matters**, and the numbers in the log are only as trustworthy
as the validation set ([Preparing training data](data.md#splitting-train-and-validation)).

### A mean IoU of about 0.41 means "everything is background"

Look at epochs 1 to 4 of the example. `loss` was falling but `val_iou` sat at 0.41. With a two-class map where the
target covers a fifth of the image, a network that predicts *background everywhere* scores IoU ≈ 0.82 on background and
0 on the target, a mean of ≈ 0.41. When you see a flat mean IoU near half the background's share, the model has not yet
learned the rare class. It is normal for the first epochs, and a warning sign if it never moves: see
[the table below](#what-to-do-when) and [Loss functions and class imbalance](losses.md).

## Checking that training works

Before you spend hours on the full dataset, prove that the pipeline can learn at all.

**Overfit ten chips.** Train on ten chips and validate on the same ten. A healthy pipeline drives the loss towards zero
and the IoU towards 1. If it cannot, the problem is in the data, the labels or the settings, not in the amount of data
or the architecture.

```python
tiny = GeoSegDataset("data/train/images", "data/train/labels", chip_size=128, max_chips_per_scene=4)
tiny._load(); tiny._chips = tiny._chips[:10]          # keep the first ten chips

cfg = TrainingConfig(output_dir="runs/overfit", num_classes=2, in_channels=4, max_epochs=30, batch_size=5,
                     learning_rate=1e-3, warmup_epochs=0, augment=False, early_stopping_patience=100,
                     num_workers=0, pin_memory=False, persistent_workers=False, devices="cpu")
res = GeoTrainer(get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False), cfg).fit(tiny, tiny)
print([round(x, 3) for x in res["history"]["train_loss"]])
```

Expected on a working setup (real run, synthetic data):

```text
Epoch 1/30  | loss=1.0211 | val_loss=0.7873 | val_iou=0.1045 | lr=1.00e-03
Epoch 13/30 | loss=0.0629 | val_loss=0.1620 | val_iou=0.8585
Epoch 30/30 | loss=0.0366 | val_loss=0.0475 | val_iou=0.9939
[1.021, 0.501, 0.314, 0.24, 0.193, 0.155, 0.131, 0.112, ..., 0.037, 0.039, 0.037]
```

(`augment=False` and a large `early_stopping_patience` keep the test about memorising; the private `_load()` call
simply loads the chips so you can slice the list.)

**Look at the predictions, not just the metric.** Run the model on a validation scene and open the result next to
the imagery and labels in a GIS. An IoU of 0.7 can hide a model that is right everywhere except the one place you care
about.

## Starting points that work

| Situation | Suggested start |
|---|---|
| A few hundred labelled chips, RGB or RGB + NIR | `unet-r50` or `segformer-b2`, `pretrained=True`, `learning_rate=1e-4`, `max_epochs=50`, default augmentation |
| Very few chips (under ~100) | `freeze_backbone=True` for the first run, then [staged fine-tuning](advanced.md#staged-fine-tuning) |
| More than 4 bands | Same, with `in_channels=<your bands>`; the extra bands start untrained, so allow more epochs |
| Rare target (under ~5% of pixels) | [A loss that handles imbalance](losses.md) |
| Object detection | `task="detection"` with `faster-rcnn-r50`; see [Models by task](../models/tasks.md#object-detection) |

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| `ImportError: albumentations is required` | `augment=True` is the default and needs the training extra | `pip install "pygeovision[train]"`, or set `augment=False` |
| `val_iou` stays near 0.41 (for a 2-class task) while `loss` falls | The model predicts only the majority class; or, in a very short run, batch-norm statistics have not settled yet | Train longer (more steps); lower the learning rate; use a [loss for imbalance](losses.md). To tell the two apart, evaluate in train mode: if the IoU is high there, the statistics just have not settled |
| `loss` falls, `val_iou` flat or falling | Over-fitting, or validation chips that look different from training | More and more varied chips, stronger augmentation, a smaller model, weight decay; check the [split](data.md#splitting-train-and-validation) |
| `loss` is `nan` | Learning rate too high, NaN in the imagery, labels outside `0…num_classes-1` | Lower `learning_rate` by 10×; run [validation](../data/validation.md) on the imagery; check label values |
| `Target N is out of bounds` | A label value ≥ `num_classes` other than 255 | Re-map labels to `0…num_classes-1`; use 255 only for "ignore" |
| Out of memory | Batch, chip or model too large | `batch_size` first, then chip size, then `precision="fp16"` / `grad_accumulation_steps`, then a smaller model |
| Training is very slow | CPU, `num_workers=0`, or tiny batches on GPU | A GPU; `num_workers=4`; check `device=` in the first log line |
| Prediction from the CLI looks wrong though `val_iou` was high | Pixel scaling differs between training and prediction | [Matching the input to training](../inference/index.md#matching-the-input-to-training) |
| `size mismatch` / `Missing key(s)` loading the checkpoint | The model name, classes or bands differ from training | [How the registries relate](../models/registry.md) |
| Old checkpoints from an earlier run are still in the folder | A new run does not clear `output_dir`; `best.pth` and `last.pth` are overwritten, older `epoch_*` files stay | Use a fresh `output_dir` per experiment |
| Training stopped early | `early_stopping_patience` epochs without improvement | Expected; raise the patience, or inspect `history` for a plateau |
