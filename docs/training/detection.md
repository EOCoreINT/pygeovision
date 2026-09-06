# Object Detection Training

```python
from pygeovision.ai.models.architectures.detection import build_retinanet
from pygeovision.training.trainer import GeoTrainer, TrainingConfig

model = build_retinanet(num_classes=3, pretrained=False)
cfg = TrainingConfig(task="detection", max_epochs=50)

trainer = GeoTrainer(model, cfg)
results = trainer.fit(train_dataset, val_dataset)
```

## What was actually missing before this was real

`GeoTrainer.fit()` previously hardcoded `CrossEntropyLoss` and
`SegmentationMetrics` unconditionally, with zero task branching —
despite the CLI already accepting a `task=[segmentation,detection]`
choice that was silently discarded (never passed into `TrainingConfig`
at all).

Real detection training needed three genuinely different things from
segmentation, all now implemented:

- A real `collate_fn` for variable-length bounding-box targets (each
  image can have a different number of boxes; the default collate
  can't stack that into one tensor).
- The correct real `torchvision` detection forward pass:
  `model(images, targets)` returns a real loss dict directly during
  training — no external `loss_fn` is called, since the model computes
  its own loss internally.
- The documented `train()`-mode + `no_grad()` pattern for computing a
  real validation loss — these models only compute losses in `train()`
  mode with real targets supplied; `eval()` mode returns predictions
  instead.

## A real, separate bug this exposed

`build_retinanet`/`build_fcos` with `pretrained=False` did not actually
prevent a network download — `torchvision`'s `weights_backbone`
parameter defaults to a pretrained ResNet50 independently of the main
`weights` argument. Fixed to set both.

## Verification

Tested end-to-end with a real `torchvision` `RetinaNet` and synthetic
bounding-box data. Segmentation and detection were run side by side on
the same code afterward to confirm neither path broke the other.

A separate bug found via this same test run: the results dict always
reported `"best_val_iou"`, even for detection, where the tracked value
is actually `val_loss` (confirmed by a test run showing
`"best_val_iou": 1.77`, which is impossible for a real, bounded-0-1
IoU). Fixed to report the real metric name and value, and only include
`best_val_iou` when it's genuinely an IoU.
