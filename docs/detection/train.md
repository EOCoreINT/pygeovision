# Train a detector

Detectors learn from **boxes**: for every training image, the position and class of each object. `GeoTrainer` trains the
torchvision-style detectors (`faster-rcnn-r50`, `mask-rcnn-r50`, `fcos-r50`, `centernet-r50`) when you set
`task="detection"`. The settings are the same as for [segmentation](../segmentation/train.md); this page covers what is
different.

## Before you start

- [ ] Boxes in **pixel coordinates** `(x1, y1, x2, y2)` of the image chip, inside the image.
- [ ] Class numbers from **1**; `0` is the background.
- [ ] Objects at least about 10 pixels wide in the chip. Smaller ones are rarely learned.
- [ ] Some chips with **no objects** (an empty `boxes` tensor of shape `(0, 4)`), so the model learns what is not an object.
- [ ] A validation set from different scenes.

## The dataset format

Each sample is `(image, target)`:

```python
target = {"boxes": torch.tensor([[x1, y1, x2, y2], ...], dtype=torch.float32),
          "labels": torch.tensor([1, 1, ...])}      # int64, numbered from 1
```

`image` is a float tensor `(bands, H, W)`. A complete dataset class (synthetic here; replace the body with the chips and
boxes you cut from your scenes):

```python
import numpy as np
import torch
from torch.utils.data import Dataset


class Boxes(Dataset):
    def __init__(self, n, seed):
        g = np.random.default_rng(seed)
        self.items = []
        for _ in range(n):
            img = g.random((4, 128, 128)).astype("float32") * 0.2
            boxes = []
            for _ in range(g.integers(1, 4)):
                x, y = g.integers(5, 90, 2)
                w, h = g.integers(15, 30, 2)
                img[:, y:y + h, x:x + w] += 0.6
                boxes.append([x, y, x + w, y + h])
            self.items.append((torch.from_numpy(img),
                               {"boxes": torch.tensor(boxes, dtype=torch.float32),
                                "labels": torch.ones(len(boxes), dtype=torch.int64)}))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
```

To turn a vector layer of building or vehicle polygons into boxes, convert each polygon to pixel coordinates with the
raster's inverse transform (`~src.transform * (x, y)`) and take its bounding box.

## Train

```python
import logging
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
train, val = Boxes(16, 0), Boxes(6, 1)

model = get_model("faster-rcnn-r50", num_classes=1, in_channels=4, pretrained=False, min_size=128, max_size=128)
cfg = TrainingConfig(
    output_dir="runs/det", task="detection", num_classes=1, in_channels=4,
    max_epochs=3, batch_size=4, learning_rate=1e-3, warmup_epochs=0, augment=False,
    num_workers=0, pin_memory=False, persistent_workers=False, devices="cpu",
)
result = GeoTrainer(model, cfg).fit(train, val)
```

Output of a real run on **synthetic** squares (16 training and 6 validation 128 × 128 chips, three epochs, a CPU,
`pretrained=False`). The log shows the shape of a run; three epochs from scratch are far too few to learn, and the low
`val_map50` says so:

```text
Starting training: 3 epochs | device=cpu | precision=auto | strategy=auto
Epoch 1/3 | loss=8.3997 | val_loss=4.5991 | val_map50=0.0022 | lr=1.00e-03
Epoch 2/3 | loss=3.3077 | val_loss=0.7924 | val_map50=0.0000 | lr=7.50e-04
Epoch 3/3 | loss=0.4410 | val_loss=0.3926 | val_map50=0.0043 | lr=2.51e-04
Training complete: 1.8 min | best=val_map50=0.0043
```

### What is different from segmentation

| Setting | For detection |
|---|---|
| `num_classes` | The number of **object** types. The background is added for you (the predictor has `num_classes + 1` outputs) |
| `min_size`, `max_size` (on `get_model`) | torchvision resizes every image so its short side is `min_size`. Set both to your chip size (128 here) so objects keep their pixel size |
| `augment` | Has no effect for detection (a warning says so); augment inside your dataset instead |
| The metric | `val_map50` (mean average precision at IoU 0.5) and `val_map50_95`; checkpoints are chosen by `val_map50` |
| The loss | Computed inside the model (classification plus box regression); you cannot pass `loss_fn` |
| `in_channels` | The model's input normalisation is widened for you, so four-band or ten-band chips work |

### Reading the metrics

`val_map50` is the area under the precision-recall curve when a predicted box counts as correct if it overlaps a true box
by at least half (IoU 0.5). 1.0 is perfect. It stays near 0 at first and then rises; a value stuck at exactly 0.0 usually
means labels start at 0, boxes are outside the image, or the learning rate is too high.

## Improve it

| Problem | Try |
|---|---|
| Misses small objects | Use finer imagery (a smaller ground sample distance), keep `min_size`/`max_size` equal to the chip size so objects are not shrunk, or try `fcos-r50` / `centernet-r50` |
| Many false positives | Add chips with no objects; raise the score threshold at prediction time |
| Overlapping objects merged | Faster R-CNN or DETR rather than CenterNet |
| Few boxes (under ~200) | `pretrained=True` and `freeze_backbone=True` first, then unfreeze ([staged fine-tuning](../training/fine-tuning.md)) |
| Outlines needed | `mask-rcnn-r50` (targets also need a `masks` tensor `(N, H, W)`) |

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `val_map50` is 0.0 every epoch | Not learning yet, labels use class `0`, or boxes outside the image | Labels from 1; check box coordinates; train longer |
| Out of memory | Batch too large | `batch_size=2`; smaller `min_size`; `precision="fp16"` on a GPU |
| `RuntimeError` about empty targets | An image with no objects has the wrong box shape | Use `torch.zeros((0, 4))` and `torch.zeros((0,), dtype=torch.int64)` |
| `AssertionError: All bounding boxes should have positive height and width` | A box with `x2 <= x1` or `y2 <= y1` | Drop or fix degenerate boxes when building the dataset |

Next: [Predict with a detector](predict.md).
