# Callbacks and metrics

Early stopping, "keep the best *k* checkpoints" and "save the last epoch" are things most runs need. `GeoTrainer` already does
all three from **settings**, so for a normal run you never write a callback. This page explains what you get for free, how to
steer it, and how to get a metric of your own when the built-in ones are not what you need.

```{admonition} The short version
:class: important
`GeoTrainer` has **no callback hooks**, and `fit(metric_fn=...)` is accepted but **not used** (a warning says so). Use the settings in
[the training configuration](config.md) for early stopping and checkpoints, and compute your own score *after* training, from `best.pth`.
The classes in `pygeovision.training.callbacks` are for a training loop **you write yourself**.
```

## What the trainer already does, by setting

| You want | Setting (see [the configuration](config.md)) | Where it ends up |
|---|---|---|
| Stop when the score stops improving | `early_stopping_patience`, `early_stopping_metric`, `early_stopping_mode` | `result["epochs_trained"]` is below `max_epochs`; the log says so |
| Keep the best *k* epochs | `save_top_k`, `checkpoint_metric`, `checkpoint_mode` | `checkpoints/epoch_XXXX_<metric>_<score>.pth` |
| A file that is always the best one | (always on) | `checkpoints/best.pth` |
| Continue an interrupted run | `save_last`, `resume` | `checkpoints/last.pth`, see [Resuming](resuming.md) |
| Watch a different metric than IoU | `early_stopping_metric="val_iou_class_1"` | any name in [the metric list](config.md#metric-names) |
| See curves | `tracker="tensorboard"`, `"mlflow"` or `"wandb"` | [Experiment tracking](../experiments/tracking.md) |

```python
from pygeovision.training import TrainingConfig

cfg = TrainingConfig(
    output_dir="runs/crops_v3", num_classes=5, in_channels=6, max_epochs=80,
    early_stopping_patience=10,                 # stop after 10 epochs without a better score
    early_stopping_metric="val_iou_class_3",    # ... of the class you actually care about
    checkpoint_metric="val_iou_class_3",        # ... and keep the epochs that are best for that class
    save_top_k=2,
)
```

## A score of your own, after training

Because `fit` ignores `metric_fn`, the way to get a custom number is a short second step:

1. train as usual;
2. load `best.pth` into the same model;
3. run the validation chips through it and score the output with `SegmentationMetrics` or your own function.

```python
import torch
from pygeovision.models import get_model
from pygeovision.training import SegmentationMetrics

model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
model.load_state_dict(torch.load("runs/crops_v3/checkpoints/best.pth", map_location="cpu")["model_state_dict"])
model.eval()

metrics = SegmentationMetrics(num_classes=3)          # ignores label 255
with torch.no_grad():
    for images, labels in val_loader:                  # your validation DataLoader
        metrics.update(model(images).argmax(1), labels)
print(metrics.compute())
```

`compute()` returns a dictionary: `mean_iou`, `mean_f1`, `accuracy`, and for each class `iou_class_0`, `f1_class_0`,
`precision_class_0`, `recall_class_0`, and so on. Label **255** is ignored, so unlabelled pixels do not count. A cost of
your own (for example "a missed building is five times worse than a false one") is a few lines on the same tensors:

```python
def weighted_error(pred, true, miss_cost=5.0):
    valid = true != 255
    missed = ((true == 1) & (pred == 0) & valid).sum()       # label says building, model says not
    false_alarm = ((true == 0) & (pred == 1) & valid).sum()
    return float(miss_cost * missed + false_alarm) / float(valid.sum())
```

```{note}
The weights are stored under the key `model_state_dict` in files written by `GeoTrainer`. (`ModelCheckpoint`, the callback class,
uses `model_state`; the two file layouts are not interchangeable. Open a file with `torch.load(path, weights_only=False)` and look at
its keys if you are unsure.)
```

## Callbacks for a loop you write yourself

Sometimes `GeoTrainer` is the wrong shape: a custom model with two heads, a GAN-style loop, a research experiment.
`pygeovision.training.callbacks` gives you the same early-stopping and checkpoint behaviour as small classes, so you do not have
to rewrite it.

| Class | What it does | Key arguments |
|---|---|---|
| `Callback` | The base class: override `on_train_start`, `on_epoch_start`, `on_epoch_end(trainer, epoch, metrics)`, `on_batch_end`, `on_train_end` | |
| `EarlyStopping` | Sets `stop_training = True` after `patience` epochs without improvement, and puts the best weights back | `monitor="val_iou"`, `patience=10`, `mode="max"`, `min_delta=0.0`, `restore_best_weights=True` |
| `ModelCheckpoint` | Writes `epoch{epoch:03d}.pth` each epoch, keeps the best `save_top_k`, and copies the latest to `last.pth` | `dirpath`, `filename`, `monitor`, `mode`, `save_top_k=3`, `save_last=True` |
| `LearningRateMonitor` | Records the learning rate of every parameter group and adds `lr` to the metrics dictionary | none |
| `ProgressBar` | Prints a one-line progress bar with the metrics and an ETA | `total_epochs` |

A callback needs only two things from the "trainer" you pass: an attribute `model` and, for checkpoints and the learning-rate
monitor, an attribute `optimizer`. Any object with those attributes will do, which is how you use the classes in your own loop.

Real output of this loop (a four-band toy rule that a two-layer network can learn; the whole script runs in a few seconds on a CPU):

```python
import types
import torch
from torch import nn
from pygeovision.training import SegmentationMetrics
from pygeovision.training.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

model = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)

trainer = types.SimpleNamespace(model=model, optimizer=optimizer)          # callbacks read only .model and .optimizer
checkpoint = ModelCheckpoint("ckpts", monitor="val_mean_iou", mode="max", save_top_k=2)
stopper = EarlyStopping(monitor="val_mean_iou", patience=3, mode="max", min_delta=0.001)
callbacks = [checkpoint, stopper, LearningRateMonitor()]

for epoch in range(1, 21):
    model.train()
    for images, labels in train_batches:                                    # your batches
        optimizer.zero_grad()
        loss_fn(model(images), labels).backward()
        optimizer.step()

    model.eval()
    metrics_obj = SegmentationMetrics(num_classes=2)
    with torch.no_grad():
        for images, labels in val_batches:
            metrics_obj.update(model(images).argmax(1), labels)
    metrics = {"val_mean_iou": float(metrics_obj.compute()["mean_iou"])}

    for cb in callbacks:
        cb.on_epoch_end(trainer, epoch, metrics)                            # you call the hooks yourself
    if stopper.stop_training:
        break

print("best checkpoint:", checkpoint.best_model_path)
```

```text
epoch  1  val_mean_iou=0.360
epoch  5  val_mean_iou=0.686
epoch  9  val_mean_iou=0.876
epoch 10  val_mean_iou=0.875
epoch 11  val_mean_iou=0.865
epoch 12  val_mean_iou=0.863
stopped at epoch 12
best checkpoint: ckpts/epoch009.pth
```

Epoch 9 was the best; epochs 10, 11 and 12 did not beat it by `min_delta`, so after `patience=3` the stopper fired and put the
epoch-9 weights back into `model`. The folder then holds `epoch009.pth`, `epoch010.pth` (the two best) and `last.pth` (epoch 12).

### Rules to know

- **Hooks are called by you.** The classes do nothing on their own; `cb.on_epoch_end(trainer, epoch, metrics)` is the call.
- **`metrics` is a dictionary** and `monitor` must be one of its keys. If it is not, `EarlyStopping` logs a warning and does nothing,
  and `ModelCheckpoint` treats the value as `0.0`, so a misspelt `monitor` quietly keeps the wrong files. Check the spelling.
- **`restore_best_weights=True`** replaces the model's weights **when the stop fires**. If your loop ends by reaching the last
  epoch instead, call `model.load_state_dict(stopper.best_weights)` yourself.
- **`last.pth` is the latest epoch**, not the best; `best_model_path` is the best of the kept files. Do not confuse them.
- **Use a different `dirpath` for every run.** The callback does not clear the folder.

## Related

- [The training configuration](config.md): every setting, including the ones above.
- [Resuming an interrupted run](resuming.md) and [Experiment tracking](../experiments/tracking.md).
- [Evaluating a model](../experiments/benchmark.md): scoring a checkpoint on data it has not seen.
