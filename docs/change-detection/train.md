# Train a change-detection network

A change network learns from **pairs of images and a change mask**. `GeoTrainer` feeds a model one tensor, so the two dates
are stacked into one multi-band image and, for networks that take two inputs, split again inside a small wrapper.

## Before you start

- [ ] Two images of the same place on **exactly the same grid** (clip and resample first).
- [ ] A change mask on that grid: `0` unchanged, `1` changed, `255` ignore.
- [ ] Pairs from different places for the validation set.
- [ ] Both dates scaled the same way.

## 1. Stack each pair

Write the two dates as one image, before bands first, and keep the mask beside it:

```python
import numpy as np
import rasterio


def stack_pair(before, after, mask, out_image, out_label):
    """Write the two dates as one 8-band image (before bands first) and copy the change mask."""
    with rasterio.open(before) as b, rasterio.open(after) as a:
        data = np.concatenate([b.read(), a.read()]).astype("float32")
        profile = b.profile
    profile.update(count=data.shape[0], dtype="float32")
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    with rasterio.open(out_image, "w", **profile) as dst:
        dst.write(data)
    with rasterio.open(mask) as m:
        meta = m.profile
        for key in ("blockxsize", "blockysize", "tiled"):
            meta.pop(key, None)
        with rasterio.open(out_label, "w", **meta) as dst:
            dst.write(m.read())
```

Then the folder layout is the same as for [segmentation](../segmentation/train.md): `data/train/images`, `data/train/labels`,
`data/val/images`, `data/val/labels`, and `GeoSegDataset` cuts chips from the stacked images.

## 2. The compact network: train it directly

`changeformer-mit-b0` accepts the stacked tensor. Build it with `in_channels` = bands **per date**:

```python
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer, TrainingConfig
from pygeovision.training.data import GeoSegDataset

train = GeoSegDataset("data/train/images", "data/train/labels", chip_size=64)
val = GeoSegDataset("data/val/images", "data/val/labels", chip_size=64)

net = get_model("changeformer-mit-b0", num_classes=2, in_channels=4, pretrained=False)    # 4 bands per date
cfg = TrainingConfig(
    output_dir="runs/cd", task="segmentation", num_classes=2, in_channels=4,
    max_epochs=4, batch_size=4, learning_rate=1e-3, warmup_epochs=0, augment=False,
    num_workers=0, pin_memory=False, persistent_workers=False, devices="cpu",
)
result = GeoTrainer(net, cfg).fit(train, val)
```

Output of a real run on **synthetic** pairs (three training and one validation 128 × 128 pair, four bands per date, the
second date brighter inside the changed square, a CPU):

```text
Epoch 1/4 | loss=0.6360 | val_loss=0.6413 | val_iou=0.4237 | lr=1.00e-03
Epoch 2/4 | loss=0.4584 | val_loss=0.4635 | val_iou=0.4490 | lr=8.54e-04
Epoch 3/4 | loss=0.2600 | val_loss=0.2820 | val_iou=0.5414 | lr=5.01e-04
Epoch 4/4 | loss=0.1803 | val_loss=0.2072 | val_iou=0.8172 | lr=1.47e-04
```

## 3. BIT and DSAMNet: wrap them

`bit` and `dsamnet` need two arguments, so wrap them so `GeoTrainer` can feed one tensor:

```python
import torch
import torch.nn as nn


class Stacked(nn.Module):
    """GeoTrainer feeds one tensor; split it into the before and after images."""
    def __init__(self, net, bands):
        super().__init__()
        self.net, self.bands = net, bands

    def forward(self, x):
        return self.net(x[:, :self.bands], x[:, self.bands:])


net = get_model("bit-r50", num_classes=2, in_channels=4, pretrained=False)
trainer = GeoTrainer(Stacked(net, 4), cfg)        # dataset items: (8-band stack, mask)
trainer.fit(train, val)
```

(Set `TrainingConfig(in_channels=4)` here too; the dataset supplies the eight stacked bands.)

### Save the inner network

```python
torch.save(net.state_dict(), "bit.pth")           # the inner network, so the CLI can load it
```

Save `net.state_dict()`, not the trainer's checkpoint: the trainer's keys carry the wrapper's `net.` prefix, which the CLI
does not expect. Then:

```bash
pygeovision ai change --before a.tif --after b.tif -o change.tif --method bit --backbone resnet50 --checkpoint bit.pth
```

## 4. Check it

- Judge by `val_iou` of the **changed class** (`iou_class_1`), not the mean: most pixels are unchanged, so a model that
  never predicts change still scores a high mean ([the same warning as in segmentation](../segmentation/train.md#read-the-per-class-values)).
- Look at the map. An under-trained network gives a confident-looking result (a two-epoch network marked 99.997% of one test
  scene as changed).
- Validate on pairs from another place and another season.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `expected input to have 8 channels, but got 4` | `in_channels` set to the stacked count | Use the bands **per date** (4), and feed the 8-band stack |
| Map is all "changed" or all "unchanged" | Class imbalance, or too little training | A [balanced loss](../training/losses.md); more pairs with change; more epochs |
| `size mismatch` when the CLI loads the weights | The wrapper's checkpoint was saved | Save `net.state_dict()` of the inner network |
| False change on every season | The training pairs do not contain seasonal differences | Add pairs from different seasons labelled "unchanged" |
