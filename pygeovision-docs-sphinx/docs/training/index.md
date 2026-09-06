# Training

`GeoTrainer` (`pygeovision.training.trainer`) is the real, live training
loop — the one the CLI's `pygeovision ai train` builds a config for.

```{warning}
`pygeovision ai train` does **not** train anything itself. It builds a
`TrainingConfig` and prints the real Python you need to run to actually
train — a real, deliberate design (dataset wiring is left to the
caller), but worth knowing up front so `--output` isn't mistaken for a
guarantee that a file will be written.
```

## A real, minimal example

```python
import segmentation_models_pytorch as smp
from pygeovision.training.trainer import GeoTrainer, TrainingConfig

model = smp.Unet(encoder_name="resnet50", in_channels=4, classes=2)
cfg = TrainingConfig(output_dir="./checkpoints", max_epochs=50)

trainer = GeoTrainer(model, cfg)
results = trainer.fit(train_dataset, val_dataset)
```

## A severe, real bug found and fixed: the default scheduler

The training loop steps the learning-rate scheduler once per batch.
The default scheduler, `CosineAnnealingLR`, was built with
`T_max=cfg.max_epochs` — treating epoch count as if it were step
count. Verified by simulation: with `max_epochs=10` and 50 steps per
epoch, this produced **50 full oscillation cycles** instead of one
smooth decay across the whole run. Since cosine is the default, every
user not explicitly overriding the scheduler was affected.

Fixed to use the real total step count, consistent with how
`"linear"`/`"onecycle"` already handled this correctly in the same
function. The same unit-mismatch bug was found and fixed in `"step"`
and the fallback branch too.

## Two real, verified metrics implementations

`SegmentationMetrics` and a real `torchmetrics`-backed `DetectionMetrics`
(delegating to a trusted library for mAP rather than a hand-rolled
implementation) are both hand-verified against known values — a
constructed confusion matrix with known TP/FP/FN counts for
segmentation, and both a perfect prediction (mAP=1.0) and a completely
missed one (mAP=0.0) for detection.

## What's real vs. dead code in the surrounding modules

`GeoTrainer` only actually uses two of the nine sibling modules in its
package (`metrics.py`, `optimizer.py`). `callbacks.py`, `checkpoint.py`,
`mixed_precision.py`, `distributed.py`, and `experiment.py` are real,
substantial code — but confirmed unused; `GeoTrainer` reimplements
simpler inline versions instead. The inline versions actually used
(its own `EarlyStopping`, `CheckpointManager`) were independently
verified correct. This is flagged rather than silently left, since
~700 lines of real, unused code in a training package is worth knowing
about even though it isn't broken.

```{toctree}
:maxdepth: 1

finetuning
detection
```
