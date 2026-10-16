# Advanced training

These are the techniques you reach for once the basic loop in [Training a model](index.md) works: saving memory,
continuing a stopped run, fine-tuning gently, using several GPUs, searching for good settings, tracking experiments, and
getting the finished model out of PyTorch. Each section says **why** you would do it, **how**, **what you should see**,
and **what to do when** it does not work.

```{admonition} What was and was not run for this page
:class: note
Everything below that shows a log was run for real on a CPU with synthetic data. **Mixed precision, multi-GPU
training, Optuna, MLflow, Weights & Biases and TensorBoard need a GPU or packages that were not available**, so those
sections describe the behaviour from the code and tell you what to check. Where that is the case it says so.
```

## Staged fine-tuning

**Why.** A pretrained encoder already knows edges, textures and shapes. A freshly created segmentation head knows
nothing, and its first updates are large and wrong. If everything trains at once, those wrong updates flow back into the
encoder and damage what it knew. Training the head first, then the encoder at a lower learning rate, protects the
pretrained knowledge. It matters most with few labels.

**How.**

```python
cfg = TrainingConfig(
    output_dir="runs/staged", num_classes=2, in_channels=4,
    learning_rate=3e-4,
    unfreeze_backbone_epoch=1,     # the encoder stays fixed for the first epoch, then joins
    backbone_lr_scale=0.1,         # and learns at one tenth of the head's rate
    max_epochs=30,
)
model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=True)
GeoTrainer(model, cfg).fit(train, val)
```

The simpler forms: `freeze_backbone=True` keeps the encoder fixed for the **whole** run (a good first experiment with
fewer than ~100 chips); `backbone_lr_scale=0.1` alone trains everything from the start with a gentler encoder.

**Expected log** (a real run, 3 epochs, `unfreeze_backbone_epoch=1`):

```text
Froze 159 backbone parameter tensor(s) for the first 1 epoch(s).
Starting training: 3 epochs | device=cpu | precision=auto | strategy=auto
Epoch 1/3 | loss=0.7557 | val_loss=0.6604 | val_iou=0.7804 | lr=3.00e-04
Epoch 2: backbone unfrozen (learning rate x0.1).
Epoch 2/3 | loss=0.5748 | val_loss=0.6163 | val_iou=0.4401 | lr=2.25e-04
```

The `lr=` shown is the head's. The score may dip when the encoder joins; watch whether it recovers over the next epochs
before deciding the unfreeze was too early (try a later epoch or a smaller `backbone_lr_scale`).

**What to do when.**

| Symptom | Fix |
|---|---|
| Warning "no parameters matched the backbone-name patterns" | Your model's encoder is not called `encoder`, `backbone` or `patch_embed`. Rename the submodule, or freeze by hand: `for p in model.my_encoder.parameters(): p.requires_grad = False` |
| Score collapses when the encoder is unfrozen | Later `unfreeze_backbone_epoch`, smaller `backbone_lr_scale` (0.01) |

## Fine-tuning from your own checkpoint

**Why.** You trained on region A and now want region B, or you trained a 2-class model and now need 5 classes. Starting
from your earlier weights beats starting from scratch.

**How.** Build the model with the **new** class count and point `checkpoint_path` at the old weights:

```python
model = get_model("unet-r50", num_classes=3, in_channels=4, pretrained=False)
cfg = TrainingConfig(output_dir="runs/ft", num_classes=3, in_channels=4,
                     checkpoint_path="runs/res/checkpoints/best.pth", learning_rate=1e-4)
GeoTrainer(model, cfg).fit(train, val)
```

The weights load **non-strictly**: layers that fit are filled in, the rest keep their fresh values. Real log when a
2-class checkpoint goes into a 3-class model:

```text
Loaded checkpoint runs/res/checkpoints/best.pth with a partial match (strict=False) -- 2 missing key(s), 0 unexpected
key(s). This is expected if finetuning into a different num_classes/head; unexpected if you intended an exact match.
missing=['segmentation_head.0.weight', 'segmentation_head.0.bias']
```

Into a model of the same shape the log says `Loaded checkpoint … (exact match).`

This is the one place where PyGeoVision tolerates a partial match, and it is on purpose: changing the head is the
point. Read the `missing=` list. If it names layers in the **middle** of the network, you loaded the wrong
architecture. If the checkpoint cannot be read at all, training stops with an error rather than starting from random
weights.

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| Out of memory | Batch, chip or model too large | `batch_size` first; then `grad_accumulation_steps` to keep the effective batch; then `fp16`/`bf16`; then smaller chips or model |
| Training is slower than expected on GPU | Data loading is the bottleneck | `num_workers` 4–8, `pin_memory=True`, `persistent_workers=True`; keep chips on a local SSD |
| Results change between identical runs | GPU kernels are not fully deterministic | Same `seed` gives similar, not identical, results; compare several seeds before concluding one setting is better |
| Validation score fluctuates wildly | Validation set too small, or small batches with batch-norm | More validation chips; larger effective batch; lower learning rate |
| `val_iou` fine but one class is poor | Averaged over classes | Early-stop and checkpoint on that class: `early_stopping_metric="val_iou_class_K"` |
