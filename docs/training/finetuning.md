# Finetuning

Two real capabilities support finetuning: freezing the backbone, and
resuming from an existing checkpoint. Both were found broken (or
entirely missing) in an earlier audit and are now fixed and verified.

## `freeze_backbone`: a real, confirmed fix

```python
cfg = TrainingConfig(freeze_backbone=True, ...)
```

This `TrainingConfig` field previously existed but was referenced
nowhere else in the training loop — setting it to `True` had zero
effect on training. It's now wired in: real backbone/encoder
parameters (matched by name pattern) are frozen before the optimizer
is built.

Verified against a real `segmentation_models_pytorch` UNet: 60 of 92
real encoder parameters correctly freeze, while the decoder remains
trainable.

```{note}
If `freeze_backbone=True` freezes nothing, it means the model's real
parameter names don't match the expected `encoder`/`backbone` name
patterns — check `model.named_parameters()` to find the real prefix
your specific architecture uses. This is now a clear, logged warning
rather than a silent no-op.
```

## `checkpoint_path`: real finetuning from an existing model

```python
cfg = TrainingConfig(checkpoint_path="./checkpoints/prior_run.pth")
```

This capability didn't exist at all before an earlier audit. It's
loaded with `strict=False` and a clear report of any mismatch — which
matters for real finetuning, since a checkpoint from a model with a
different number of output classes will have a differently-shaped
final layer.

Verified against both an exact-match case (0 missing/unexpected keys)
and a realistic finetuning case: a checkpoint trained for 2 classes,
loaded into a 5-class model. The real, valuable encoder weights
transfer correctly; only the differently-shaped head is left to train
from scratch — exactly the intended finetuning behavior.

```{warning}
`CheckpointManager.save()` never saves optimizer state, only model
weights. This is fine for finetuning (which typically wants a fresh
optimizer on new data anyway) but means `checkpoint_path` does not
support fully resuming interrupted training with optimizer momentum
intact.
```
