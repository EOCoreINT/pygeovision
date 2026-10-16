# Train a segmentation model

This page is the segmentation-specific guide to training: what the labels must look like, which settings matter, how to
read the result and how to test it. The general mechanics (the loop, the log, checkpoints) are in
[Training a model](../training/index.md); the full list of settings is in [The training configuration](../training/config.md).

## Before you start

- [ ] Images and labels are on the **same pixel grid** (same CRS, same transform, same size).
- [ ] Labels are a single-band `uint8` raster: `0` = background, `1..N-1` = your classes, `255` = *ignore* (never counted).
- [ ] You have a **separate validation set**, ideally from a different area or date than the training chips.
- [ ] You know your `in_channels` (bands) and `num_classes` (**background counts**).
- [ ] The pixel values are scaled the way you will scale them at prediction time ([why](../inference/index.md#matching-the-input-to-training)).

## 1. Labels and chips

Cut scenes into chips and split them without leaking neighbouring chips across the split:
[Preparing training data](../training/data.md). `GeoSegDataset` does the cutting and pairs every image chip with its
label chip.

```python
from pygeovision.training.data import GeoSegDataset

train = GeoSegDataset("data/train/images", "data/train/labels", chip_size=128)
val   = GeoSegDataset("data/val/images",   "data/val/labels",   chip_size=128)
print("chips:", len(train), len(val))
```

Rules of thumb:

| Question | Guidance |
|---|---|
| Chip size | 128 or 256 pixels. Larger chips give more context but fewer samples per batch and more memory. Must be divisible by 32 for the CNN models |
| How many chips | Hundreds can work with pretrained weights; thousands are better. Chips with **none** of your target are useful too, they teach the model what is not the target |
| Class balance | If a class is under about 5% of pixels, use a [balanced loss](../training/losses.md) and read per-class scores |

## 2. Pick a model and a starting point

| Your data | Start with |
|---|---|
| Under ~100 chips | `unet-r50`, `pretrained=True`, `freeze_backbone=True` for the first run, then [staged fine-tuning](../training/fine-tuning.md#staged-fine-tuning) |
| A few hundred to a few thousand chips | `unet-r50` or `segformer-b2`, `pretrained=True`, `learning_rate=1e-4`, `max_epochs=50` |
| Tens of thousands, a large GPU | `segformer-b5` or `upernet-swin-b` |
| Multispectral time series, few labels | A foundation model encoder: [Prithvi-EO](../foundation-models/prithvi.md) |
| No GPU | `segformer-b0` or `unet-efficientb4` at chip size 128 |

All models: [Segmentation models](models.md).

## 3. Train

```python
import logging
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")

model = get_model("unet-r50", num_classes=2, in_channels=4, pretrained=False)
cfg = TrainingConfig(
    output_dir="runs/seg", task="segmentation", num_classes=2, in_channels=4,
    max_epochs=4, batch_size=4, learning_rate=3e-4, warmup_epochs=0,
    num_workers=0, pin_memory=False, persistent_workers=False, devices="cpu", augment=False,
)
result = GeoTrainer(model, cfg).fit(train, val)
print(result["best_checkpoint"], result["best_metric_name"], round(result["best_metric_value"], 4))
```

Output of a real run on **synthetic** data (three 128 × 128 four-band scenes with a square as the target, a CPU,
`pretrained=False`). It shows what the log looks like; it says nothing about accuracy on your imagery:

```text
Starting training: 4 epochs | device=cpu | precision=auto | strategy=auto
Epoch 1/4 | loss=0.7209 | val_loss=0.6603 | val_iou=0.5672 | lr=3.00e-04
Epoch 2/4 | loss=0.4996 | val_loss=0.5861 | val_iou=0.8157 | lr=2.56e-04
Epoch 3/4 | loss=0.4314 | val_loss=0.5387 | val_iou=0.8664 | lr=1.50e-04
Epoch 4/4 | loss=0.3904 | val_loss=0.5012 | val_iou=0.8651 | lr=4.48e-05
Training complete: 0.2 min | best=val_iou=0.8664
runs/seg/checkpoints/epoch_0003_val_iou_0.8664.pth val_iou 0.8664
```

### The settings that matter for segmentation

| Setting | What it does | Typical change |
|---|---|---|
| `num_classes`, `in_channels` | Must match labels and bands | Always set |
| `pretrained` (on `get_model`) | Start from ImageNet weights | `True` with real data |
| `learning_rate` | Step size | `1e-4` fine-tuning; `3e-4` from scratch |
| `freeze_backbone`, `unfreeze_backbone_epoch`, `backbone_lr_scale` | Train the head first, then the encoder gently | [Staged fine-tuning](../training/fine-tuning.md#staged-fine-tuning) |
| `augment`, `augment_mode` | Flips, rotations, brightness (needs `albumentations`). `light` … `heavy` | `medium`; `geometric_only` if brightness carries meaning |
| `early_stopping_patience`, `early_stopping_metric` | Stop when validation stops improving | 10 to 15 epochs on `val_iou` |
| `precision` | `fp16` or `bf16` saves memory on a GPU | [Speed and memory](../training/performance.md) |

## 4. Read the log

- `loss` is the training loss, `val_loss` and `val_iou` are measured on the validation set. **Judge by the validation
  numbers.** If `loss` falls while `val_loss` rises, the model is memorising ([over-fitting](../training/index.md#reading-the-training-log)).
- A mean IoU near 0.4 with two classes usually means "everything is background": see
  [the explanation](../training/index.md#a-mean-iou-of-about-041-means-everything-is-background).
- Keep `best.pth`: it is the epoch with the best validation score, and the file to deploy.

## 5. Test it

The validation score during training chooses the checkpoint, so it is optimistic. Report accuracy on a **third** set the
training never saw. With `SegmentationMetrics` you can compute the same figures yourself:

```python
import torch
from pygeovision.training import SegmentationMetrics

state = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=True)
model.load_state_dict(state["model_state_dict"]); model.eval()

metrics = SegmentationMetrics(num_classes=2)          # 255 in the reference is ignored
with torch.no_grad():
    for x, y in torch.utils.data.DataLoader(val, batch_size=4):
        metrics.update(model(x).argmax(1), y)
scores = metrics.compute()
print({k: round(v, 3) for k, v in scores.items() if k in ("mean_iou", "mean_f1", "accuracy", "iou_class_0", "iou_class_1")})
```

```text
{'mean_iou': 0.866, 'mean_f1': 0.926, 'accuracy': 0.961, 'iou_class_0': 0.955, 'iou_class_1': 0.778}
```

(The same synthetic run. `compute()` also returns `precision_class_N` and `recall_class_N` for every class.)

### Read the per-class values

The mean over classes hides rare classes. A model that finds none of a rare class can still have a high mean because
the background is easy. Look at `iou_class_N`, `precision_class_N` and `recall_class_N` for each class: low recall means
the class is missed, low precision means it is over-predicted.

## 6. Use it

Predict a scene: [Predict with a segmentation model](predict.md). Explain why the model decided: [Explainability](../explainability/index.md).
Check the model in production: [Monitoring](../monitoring/index.md).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `val_iou` stuck near 0.4 | The model predicts only background | A [balanced loss](../training/losses.md); more chips containing the target; check that label values are 0 and 1, not 0 and 255 |
| `val_iou` high, map looks wrong | Validation chips overlap training chips (leakage) or the scaling differs at prediction | Split by area ([data](../training/data.md)); use the same scaling |
| Loss is `nan` | Learning rate too high, or `nan` in the input | Lower `learning_rate`; replace `nan` pixels in the input before training; `grad_clip` is on by default |
| Out of memory | Chips or batch too large | Lower `batch_size`, use `precision="fp16"` or accumulate gradients ([speed and memory](../training/performance.md)) |
| `size mismatch` when loading | Model name, classes or bands differ from training | Build the model exactly as in training |
