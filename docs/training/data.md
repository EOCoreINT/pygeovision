# Preparing training data

Most failed training runs are data problems that no setting can fix: labels shifted by a few pixels, class numbers that
skip a value, a validation area that sits next to the training area, imagery scaled one way for training and another
way for prediction. This page covers each of these, in the order you meet them.

## What a training set is

A **chip** is a small square piece of a scene, for example 256 × 256 pixels. A training set is a collection of chips in
pairs:

| Part | What it is | Format |
|---|---|---|
| **Image chip** | The bands your model will see | GeoTIFF, `float32`, one band per input channel |
| **Label chip** | The answer for every pixel | GeoTIFF, one band, integer class numbers |

Both files of a pair have the **same name** in two folders, and both are on the **same pixel grid**: pixel (row 10,
column 20) in the image is the same piece of ground as pixel (10, 20) in the label.

```text
data/
├── train/
│   ├── images/   a_0_0.tif  a_0_128.tif  b_0_0.tif ...
│   └── labels/   a_0_0.tif  a_0_128.tif  b_0_0.tif ...
└── val/
    ├── images/   c_0_0.tif ...
    └── labels/   c_0_0.tif ...
```

## Rules for labels

| Rule | Why | Check |
|---|---|---|
| **Class numbers are `0, 1, 2 … N-1` with no gaps**, and `num_classes = N` | The network has one output per class number; a label of 7 in a 3-class model has no output to match | `np.unique(label)` |
| **Background is a class** (usually 0) if you want it predicted | A two-class water map has classes 0 (not water) and 1 (water), so `num_classes=2` | Count it |
| **255 means "ignore this pixel"** | Unlabelled or uncertain pixels should not teach the model anything. The trainer's loss and metrics skip 255 | Use 255, not a class number, for "unknown" |
| **Binary masks are 0/1, not 0/255** | A `0/255` mask is read as classes 0 and 255; 255 is then ignored and the target is never learned | `np.unique(label)` must not show only `[0, 255]` |
| **`uint8` dtype** is enough | Labels are read as integers; 255 fits | `rasterio` shows the dtype |

```python
import numpy as np, rasterio
with rasterio.open("labels/a_0_0.tif") as src:
    print(src.dtypes[0], np.unique(src.read(1), return_counts=True))
# uint8 (array([  0,   1, 255], dtype=uint8), array([40211, 21480,  3845]))
```

If your labels came from a vector layer, rasterize them onto the image's grid (burn class numbers; fill unknown areas
with 255). If they came from a published map, resample them to the image grid with nearest-neighbour (never bilinear,
which invents in-between class numbers). [Labeling](../labeling/index.md) covers both.

```{admonition} Sanity-check the class balance before training
:class: tip
Add up the pixels per class over the whole training set. If one class is under about 5% of the labelled pixels, plan
for it now: see [Loss functions and class imbalance](losses.md).
```

## Rules for images

| Rule | Why |
|---|---|
| **Same band order and count in every scene** | The first layer learns "channel 0 is blue". A scene with a different order silently breaks it |
| **No `NaN` or `Inf`** | One bad pixel makes the loss `nan` for the whole run. [Validation](../data/validation.md) finds them |
| **A sensible, consistent value range** | See [Scaling the pixel values](#scaling-the-pixel-values) below: this is the most common hidden error |
| **The same pixel size** across scenes (10 m with 10 m, not 10 m with 30 m) | The network learns shapes at a scale. [Preprocessing](../data/preprocessing.md) resamples |
| **Clouds handled** | A cloud labelled "forest" teaches the model that white means forest. Mask clouds or mark them 255 in the label |

[AI-ready data](../data/ai-ready.md) wraps the checks and the scaling in one call, `client.prepare_for_ai`.

## Cutting chips

Whole scenes are too big to train on, so they are cut into tiles. The command does it for one image and its label:

```bash
pygeovision ai chips --image scene.tif --label scene_labels.tif --output ./data/all --chip-size 256 --prefix a
```

```text
✓ 16 chips exported → ./data/all
```

| Option | Default | Meaning |
|---|---|---|
| `--image`, `--label` | required | The imagery and its label raster. **They must be on the same grid**: same size, same CRS, same origin. If not, the command stops with *"image and label are not on the same grid"*; resample the label onto the image first |
| `--output` | required | Writes `images/` and `labels/` folders under it |
| `--chip-size` | 256 | Tile edge in pixels. 256 is a good start; 128 trains faster and uses less memory, 512 gives more context |
| `--overlap` | 0 | Pixels shared between neighbouring chips. Overlap gives more (correlated) chips and helps objects cut by tile edges; it also raises the risk of leakage between train and validation, see below |
| `--prefix` | `chip` | Start of every file name. Use a different prefix for each scene when they share one output folder, otherwise the second scene overwrites the first |

What it does and does not do:

- Files are named `<prefix>_<row>_<col>.tif`, where row and column are the pixel offset of the chip's corner. Each keeps
  its georeferencing.
- Strips at the right and bottom edge that are narrower than a chip are **dropped**, not padded.
- It keeps **every** chip, including ones that are entirely background or entirely nodata. Remove the useless ones
  yourself, or let `GeoSegDataset` skip the mostly empty ones (`skip_nodata`, below).
- It does **not** split into train and validation. Do that next.

## Splitting train and validation

The validation set exists to answer one question: *how will the model do on places it has not seen?* That only works if
validation chips come from **different ground** than the training chips.

```{admonition} Do not split chips at random
:class: warning
Neighbouring chips from one scene share fields, shadows, roads and light. If a random 20% of chips is held out, each
held-out chip has a near-twin in the training data, and the validation score is flattering. The model then fails on a
genuinely new area and the log gave no warning.
```

Better splits, in order of how honest they are:

| Split | How |
|---|---|
| **By scene or region** | Train on some scenes, validate on others. Best, especially if the scenes are from different places or dates |
| **By block** within one scene | Left or top part to train, right or bottom part to validate, with a **gap** of unused chips between to stop leakage |
| **Random chips** | Only for a quick technical test. Do not report the score |

A block split for chips made by `ai chips` (train on the left, validate on the right, leave a gap of one chip column):

```python
import re, shutil
from pathlib import Path

def split_chips_by_column(chips_dir, out_dir, val_fraction=0.2, gap_chips=1):
    """Left of the scene -> train, right -> val, with unused chips in between."""
    chips_dir, out_dir = Path(chips_dir), Path(out_dir)
    pat = re.compile(r".+_(\d+)_(\d+)\.tif")
    files = sorted((chips_dir / "images").glob("*.tif"))
    cols = sorted({int(pat.fullmatch(p.name).group(2)) for p in files})
    n_val = max(1, round(len(cols) * val_fraction))
    val_cols = set(cols[-n_val:])
    gap_cols = set(cols[-n_val - gap_chips:-n_val]) if gap_chips else set()
    counts = {"train": 0, "val": 0, "dropped": 0}
    for img in files:
        col = int(pat.fullmatch(img.name).group(2))
        if col in gap_cols:
            counts["dropped"] += 1
            continue
        split = "val" if col in val_cols else "train"
        for kind in ("images", "labels"):
            dst = out_dir / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chips_dir / kind / img.name, dst / img.name)
        counts[split] += 1
    return counts

print(split_chips_by_column("data/all", "data"))
# {'train': 8, 'val': 4, 'dropped': 4}      (a 512 x 512 scene, 128-pixel chips: 4 columns of 4 chips)
```

A gap of at least one chip width (more for a network with a large receptive field) keeps the two sides apart.

How much to hold out: about 15 to 25% of the labelled area. Too small a validation set makes the score jump from
epoch to epoch (a handful of chips is not a measurement). If the data is scarce, the final model can be retrained on
train plus validation once you have chosen the settings.

## Scaling the pixel values

A network is trained on numbers, and it will only work on numbers scaled the same way. Raw satellite values (Sentinel-2
reflectance stored as 0-10000, Landsat as 0-65535, SAR in decibels) are far from what a freshly initialised network
expects, so you scale them. **How** you scale matters less than doing it **identically** at training and prediction.

| Approach | Pixel values | Good for | Watch out |
|---|---|---|---|
| **Fixed scale factor** | Reflectance ÷ 10000, clipped to 0-1 | Models that must work on many scenes and dates: a given surface always gets the same number | Needs a known scale; haze and cloud keep their bright values |
| **Per-scene percentile stretch** | Each band mapped so its 2nd percentile is 0 and 98th is 1 | Visual-style imagery, quick experiments | The same surface gets different numbers in different scenes |
| **Z-score** | Subtract the mean, divide by the standard deviation | Radar and index stacks | Mean and deviation must come from training data and be reused at prediction |

`client.prepare_for_ai(..., normalise="scale_factor" | "percentile" | "zscore" | "minmax")` writes a scaled copy of a
scene ([AI-ready data](../data/ai-ready.md)). Run it on every scene **before** cutting chips.

```{admonition} The command-line prediction stretches by default
:class: warning
`pygeovision infer predict` (and `infer batch`, `ai infer`) applies a per-band 2nd-98th percentile stretch to the scene
**before** the model sees it, because that suits imagery in raw digital numbers. If you trained on reflectance already
scaled to 0-1, add **`--no-normalise`**. Skipping this is the single most common reason for a good validation score and
a poor map. A test of this: a model that scored 0.98 validation IoU agreed with the truth on 82% of pixels (nearly all
"background") with the default stretch, and on 99.6% with `--no-normalise`. See
[Inference](../inference/index.md#matching-the-input-to-training).
```

Pick one approach for the whole project and write it down next to the checkpoint. The checkpoint stores weights, not
how you scaled the data.

## `GeoSegDataset`

`GeoSegDataset` is the class that turns the two folders into something PyTorch can train on.

```python
from pygeovision.training.data import GeoSegDataset

train = GeoSegDataset(
    "data/train/images", "data/train/labels",
    chip_size=256,                 # tile edge; scenes larger than this are cut up
    overlap=0,                     # pixels shared between neighbouring tiles
    max_chips_per_scene=None,      # cap per scene (None = all)
    skip_nodata=True,              # drop tiles that are more than half zeros
    augment=False,                 # see the note below
)
print(len(train))                  # number of chips
image, label = train[0]
print(image.shape, image.dtype, label.shape, label.dtype)
# torch.Size([4, 256, 256]) torch.float32 torch.Size([256, 256]) torch.int64
```

| Argument | Default | Meaning |
|---|---|---|
| `image_dir`, `label_dir` | required | Folders of GeoTIFFs. Files are paired by name (`s0.tif` with `s0.tif`; `.tiff` is accepted for labels). An image with no label is skipped with a warning |
| `chip_size` | 256 | Tile edge in pixels. A scene is read whole and tiled; edge strips smaller than a tile are dropped |
| `overlap` | 0 | Pixels shared by neighbouring tiles |
| `max_chips_per_scene` | `None` | Take at most this many tiles from each scene, row by row. Useful when one huge scene would otherwise dominate |
| `skip_nodata` | `True` | Skip a tile if more than half of its pixels are exactly zero in **every** band |
| `augment` | `False` | Random flips, 90° turns and a little noise on each load. See below |
| `transform` | `None` | Your own function `(image, label) -> (image, label)` applied last |

Things worth knowing:

- **Values are used as stored.** No scaling is applied. The label is read as the first band, as integers.
- **The grid is checked.** An image and label that differ in size, CRS or origin are skipped with a warning naming
  the file ("not on the same grid"). A shifted label would otherwise train quietly on wrong answers.
- **Everything is held in memory.** Chips are cut when the dataset is first used and kept as arrays. The memory is
  about `chips × bands × chip_size² × 4` bytes: 2,000 chips of 6 bands at 256 × 256 is about 3 GB. If that is too
  much, use `max_chips_per_scene`, smaller chips, or write your own dataset that reads chips from disk on demand
  (below).
- **Two places can augment.** `GeoSegDataset(augment=True)` flips, rotates and adds noise, and clips the result to
  0-1. `TrainingConfig(augment=True)` (the default) applies a separate, richer pipeline to the training set inside
  `GeoTrainer` ([configuration](config.md#augmentation)). Use one of them, not both. Both assume values between 0
  and 1: with other ranges (z-scores, decibels) set both to `False` and augment in your own `transform`.

### Writing your own dataset

`GeoTrainer` accepts any PyTorch dataset that returns `(image, label)`: an image tensor `(bands, H, W)` of `float32`
and a label tensor `(H, W)` of `int64` (long). For large data, read from disk per item:

```python
import numpy as np, rasterio, torch
from pathlib import Path

class DiskChips(torch.utils.data.Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root, "images").glob("*.tif"))
        self.label_dir = Path(root, "labels")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with rasterio.open(self.files[i]) as s:
            image = s.read().astype("float32")
        with rasterio.open(self.label_dir / self.files[i].name) as s:
            label = s.read(1).astype("int64")
        return torch.from_numpy(image), torch.from_numpy(label)
```

For detection the dataset returns `(image, {"boxes": FloatTensor[N, 4] in pixels (x1, y1, x2, y2), "labels":
LongTensor[N]})` with class numbers **starting at 1** (0 is the background of the detector); see
[Models by task](../detection/index.md#object-detection).

## `make_geo_loaders`

A shortcut that builds ready data loaders from one pair of folders:

```python
from pygeovision.training.data import make_geo_loaders

train_loader, val_loader = make_geo_loaders("data/images", "data/labels", chip_size=256, batch_size=8, split=0.8)
```

It splits the **chips at random** (80/20, fixed seed), which is exactly the leakage described [above](#splitting-train-and-validation).
Use it to try things out. For a result you will report, split by area and pass two datasets to `fit`.

## What to do when…

| Symptom | Likely cause | What to do |
|---|---|---|
| `len(dataset)` is 0 | Folders empty or names differ between images and labels; every chip over half zeros; label on another grid | Check the folder paths and file names; read the warnings the dataset logs (turn on `logging.basicConfig(level=logging.INFO)`) |
| Warning "Skipping X: image … and label … are not on the same grid" | The label raster has another size, CRS or origin | Resample the label onto the image grid (nearest-neighbour), then re-cut |
| `ai chips` says "not on the same grid" | Same | Same |
| Training works but the target is never predicted | Binary mask stored as 0/255 | Convert to 0/1 |
| `Target 7 is out of bounds` | A class number ≥ `num_classes` | Re-map to `0…N-1`; use 255 only for "ignore" |
| Validation score is very high, the real map is poor | Leakage between train and validation, or scaling differs at prediction | [Split by area](#splitting-train-and-validation); see [Inference](../inference/index.md#matching-the-input-to-training) |
| Out of memory while building the dataset | Everything is held in RAM | `max_chips_per_scene`, smaller chips, or a disk-reading dataset |
| Chips of the second scene replaced the first | Same file names | Use `--prefix` per scene |
| Chips look empty or striped | The source scene has nodata or scan gaps | Mask or fill first; keep `skip_nodata=True` |
