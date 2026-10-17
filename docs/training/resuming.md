# Resuming an interrupted run

A job ran out of time, the machine rebooted, you pressed Ctrl-C. Resuming continues from the **last finished epoch** with the same
weights, optimiser, learning-rate schedule, history and early-stopping counter, instead of starting over. Every claim on this page was
checked by running real training (CPU, a tiny model) and reading the files and logs it produced.

```{figure} ../_static/img/flow-resume.svg
:alt: Resuming a run: each finished epoch writes last.pth atomically; after an interruption the same script with resume=True restores weights, optimiser, schedule, history and early-stopping state and continues from the next epoch
:width: 100%
:class: pgv-figure

The safe point is the end of an epoch. Work inside an unfinished epoch is lost.
```

## How to resume

Run the **same** script again with `resume=True` and the **same** `output_dir`:

```python
cfg = TrainingConfig(output_dir="runs/res", max_epochs=50, resume=True, ...)
GeoTrainer(model, cfg).fit(train_set, val_set)
```

Real log of a run with `max_epochs=4` that had been stopped after two epochs:

```text
Resuming from runs/res/checkpoints/last.pth after epoch 2.
Starting training: 4 epochs | device=cpu | precision=auto | strategy=auto
Epoch 3/4 | ...
```

The result reports `resumed_from_epoch: 2` and `history` covers all four epochs (the two before the interruption and the two after). A
fresh run has `resumed_from_epoch: None`.

## What is saved, and when

After **every** finished epoch (when `save_last=True`, the default) the trainer writes `<output_dir>/checkpoints/last.pth`. It is written to
`last.pth.tmp` first and then renamed over the old file, so a crash **while saving** leaves the previous `last.pth` intact, never a half-written
one. Contents, read from a real file:

| Key | Holds |
|---|---|
| `epoch` | The number of the finished epoch |
| `model_state_dict` | The weights |
| `optimizer_state_dict` | The optimiser's momentum and other state |
| `scheduler_state_dict` | The learning-rate schedule's position (`None` when there is no schedule) |
| `scaler_state_dict` | The mixed-precision scaler (`None` on CPU or in full precision) |
| `history` | Every metric of every finished epoch (`train_loss`, `val_loss`, `val_iou`, `lr`, per-class values) |
| `global_step` | The number of optimiser steps so far |
| `early_stopping` | The best value, the patience counter and whether it had triggered |
| `checkpoints` | The list of best-k weight files, so the best-k set carries on |
| `metrics`, `max_epochs` | The last epoch's metrics and the planned length |

It sits next to the other files of the run (here with `save_top_k=2`):

```text
runs/res/checkpoints/
    best.pth                       a copy of the best epoch, a stable name for inference
    epoch_0001_val_iou_0.2500.pth  the best two epochs by the checkpoint metric (weights and metrics only)
    epoch_0002_val_iou_0.2500.pth
    last.pth                       the full training state: the only file resume reads
```

`best.pth` and `epoch_*.pth` hold `epoch`, `model_state_dict`, `metrics`, `metric` and `score`. They do **not** hold the optimiser, so they are for
inference and fine-tuning, not for resuming.

## What resuming does not restore

- **The random-number state.** The data order and augmentation after resuming are not the ones an uninterrupted run would have used. A
  resumed run is equivalent in distribution, not bit for bit.
- **Work inside the unfinished epoch.** Up to one epoch of computing is lost, which on a very large dataset can be hours.
  There is no mid-epoch checkpoint.
- **Anything not in the settings.** The script must build the same model and the same datasets. The file stores weights, not code.

## Same settings, same model

Resuming checks what it can and **refuses** rather than carrying on with something different:

| What differs | What happens |
|---|---|
| The model architecture or number of classes | `RuntimeError: resume=True but .../last.pth could not be restored into this model: ... size mismatch for weight ...`. Nothing is trained |
| A different **optimiser type** (AdamW then SGD) | `RuntimeError: Cannot resume from .../last.pth: it was written with the AdamW optimizer but this run builds SGD.` (checked before anything is loaded; files written by older versions carry no optimiser name and skip this check) |
| The staged-unfreezing epoch or the backbone learning-rate scale (anything that changes the parameter groups) | `RuntimeError: Could not restore optimizer/scheduler state ... Use the original settings to resume, or resume=False` |
| `max_epochs` | A **warning**, and the run continues with an odd learning-rate curve (below) |
| `last.pth` holds only weights | A warning; training starts at epoch 1 from those weights |
| `last.pth` does not exist | A fresh run, with the log line `resume=True but .../last.pth does not exist; starting a fresh run.` |

```{admonition} Do not change max_epochs when resuming
:class: warning
The learning-rate schedule is stored in `last.pth` with the length it was created for. Resuming a 2-epoch run with
`max_epochs=4` gave `lr=1.00e-06` at epoch 3 (the old schedule had already finished) and then `lr=1.50e-04` at epoch 4
as the cosine curve started over. A warning is logged when the numbers differ. To train **longer** than planned, start
a new run from the best weights instead: `checkpoint_path=".../best.pth"`. That is a fine-tune with a fresh optimiser and schedule.
```

## Edge cases (all run)

| Situation | Result |
|---|---|
| Resume a run that **finished** with the same `max_epochs` | `Resuming ... after epoch 3.` and nothing more is trained. `epochs_trained: 3`, `resumed_from_epoch: 3`, and `history` still has 3 entries |
| Resume a run that **stopped early** | The early-stopping state is restored as "already triggered", so **one more epoch is trained** and then it stops again (stopped at epoch 2; the resumed run trained epoch 3 and stopped). To keep training, start a new run from the best weights with `checkpoint_path` and a larger `early_stopping_patience` |
| Resume after training with `save_last=False` | There is no `last.pth`: `starting a fresh run`, `resumed_from_epoch: None`. Turn `save_last` on before a long run |
| Resume with a different `output_dir` | The same: no `last.pth` there, a fresh run |
| Resume on a different machine | Works if the folder (with `last.pth`) is copied and the same packages are present. The file is read on the CPU first, then moved to the training device |

## Resume versus fine-tune

| | `resume=True` | `checkpoint_path=...` |
|---|---|---|
| Starts from | `last.pth` of this `output_dir` | Any `.pth` that holds `model_state_dict` (`best.pth`, an earlier run) |
| Optimiser and schedule | Restored | **Fresh** |
| Epoch counter and history | Continued | Start at 1 |
| Use for | Finishing the same run | A new run from good weights: longer training, new data, other classes |

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError … could not be restored into this model` | The model or settings differ from the run that wrote `last.pth` (architecture, classes, in_channels) | Rebuild exactly the same, or set `resume=False` and use `checkpoint_path` |
| `Cannot resume ...: written with the AdamW optimizer but this run builds SGD` | The optimiser setting changed | Use the original, or start a new run |
| `Could not restore optimizer/scheduler state` | A change to `unfreeze_backbone_epoch` or `backbone_lr_scale` | Use the original settings, or start a new run |
| Resume starts from epoch 0 | `last.pth` is missing: wrong `output_dir`, or `save_last=False` earlier | Point to the right folder; keep `save_last=True` |
| The learning rate looks wrong after resuming | `max_epochs` changed | Restore the original value; see the warning above |
| The run stops one epoch after resuming | It had stopped early | See the edge-case table |
| `last.pth` is larger than `best.pth` | It also holds the optimiser (a real run: 30.5 KB against 10.9 KB for AdamW, 20.8 KB for SGD with momentum) | Delete it after the run is finished; it is only needed to resume |

## Related

- [Training configuration: checkpoints and resuming](config.md#checkpoints-and-resuming), [Callbacks](callbacks.md), [Experiment tracking](../experiments/tracking.md)
- [Fine-tuning from a checkpoint](fine-tuning.md), [Export and shrinking a model](export.md)
