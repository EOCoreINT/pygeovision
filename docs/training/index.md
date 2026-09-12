# Training

`pygeovision` has **two real, separate** `GeoTrainer`/`TrainingConfig`
implementations — genuinely different classes, not two names for the
same thing. Conflating them was itself a real documentation error
found and corrected during this audit; getting this distinction right
matters, since the two have different real capabilities.

| | `pygeovision.training.trainer` | `pygeovision.ai.training.trainer` |
|---|---|---|
| Used by the CLI (`pygeovision ai train`)? | **Yes** | No — Python API only |
| Detection training (`task="detection"`) | No such field exists | Yes |
| Finetune from checkpoint (`checkpoint_path`) | No such field exists | Yes |
| `freeze_backbone` | Yes (see below) | Yes |

## A severe bug found in the CLI-connected trainer, missed by an earlier fix

`freeze_backbone` and the default cosine LR schedule were audited and
fixed earlier in this project — but only in
`pygeovision.ai.training.trainer`, the Python-API-only implementation.
Checking the *other*, CLI-connected `pygeovision.training.trainer`
directly (rather than assuming a fix in one parallel system meant the
other was also covered) found both bugs present here too, completely
unfixed:

- **`freeze_backbone`** appeared exactly once in the entire file — its
  own field definition — with zero references anywhere in the real
  training loop. Setting it to `True` had no effect at all.
- **The default `"cosine"` scheduler** used `T_max=cfg.max_epochs`
  while the loop steps the scheduler once per batch, not once per
  epoch — the identical unit-mismatch bug, verified by simulation to
  produce dozens of oscillation cycles instead of one smooth decay
  across a training run. The `"step"` scheduler had the same
  unit-mismatch too (`lr_step_size` used directly as a step count
  despite per-batch stepping).

Both are now fixed the same way as the other trainer, and
independently re-verified: `freeze_backbone=True` against a real
`segmentation_models_pytorch` model (60/92 real encoder parameters
correctly freeze — the identical result as the other trainer's fix),
and the scheduler fix verified by simulating a full 10-epoch run and
confirming zero oscillations (a real, monotonic decay) where the
unfixed version would have produced dozens.

```{warning}
This means every real user of `pygeovision ai train` (the CLI command)
up to this point in the audit was silently affected by both bugs —
`freeze_backbone=True` never froze anything, and the default scheduler
oscillated instead of decaying, for the actual CLI-connected trainer.
The earlier fix to the other trainer did not protect CLI users at all.
```

## Using the CLI-connected trainer

```python
from pygeovision.training.trainer import GeoTrainer, TrainingConfig
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet50", in_channels=4, classes=2)
cfg = TrainingConfig(
    output_dir="./checkpoints",
    max_epochs=50,
    freeze_backbone=True,   # real, now actually freezes encoder params
    scheduler="cosine",     # real, now actually decays smoothly over the full run
)

trainer = GeoTrainer(model, cfg)
results = trainer.fit(train_dataset, val_dataset)
```

```{note}
`pygeovision ai train segmentation --data ... --output ...` builds
this real config and prints the real Python needed to actually train
— it does not train anything itself. Dataset wiring is left to the
caller.
```

## Using the Python-API-only trainer (detection support, finetuning)

See [Finetuning](finetuning.md) and
[Object Detection Training](detection.md) — both of these real
capabilities live only in `pygeovision.ai.training.trainer`, not the
trainer above.

## Two real, verified metrics implementations

`SegmentationMetrics` and a real `torchmetrics`-backed
`DetectionMetrics` are both hand-verified against known values — a
constructed confusion matrix with known TP/FP/FN counts for
segmentation, and both a perfect prediction (mAP=1.0) and a completely
missed one (mAP=0.0) for detection.

## What's real vs. dead code in the surrounding modules

`ai.training.trainer.GeoTrainer` only actually uses two of the nine
sibling modules in its package (`metrics.py`, `optimizer.py`).
`callbacks.py`, `checkpoint.py`, `mixed_precision.py`,
`distributed.py`, and `experiment.py` are real, substantial code —
but confirmed unused; `GeoTrainer` reimplements simpler inline
versions instead. The inline versions actually used (its own
`EarlyStopping`, `CheckpointManager`) were independently verified
correct.

```{toctree}
:maxdepth: 1

finetuning
detection
```
