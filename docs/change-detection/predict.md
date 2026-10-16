# Predict change

## Without training: change vector analysis

`spectral_diff` is the default method. For every pixel it measures the length of the band-difference vector between the
two dates, then splits "changed" from "unchanged" with an automatic Otsu threshold.

```bash
pygeovision ai change --before 2020.tif --after 2024.tif -o change.tif
```

```text
Running spectral_diff change detection...
✓ Change detection complete → change.tif
  threshold: 0.001367
  change_fraction: 0.0427
  valid_fraction: 1.0
```

| Output | Meaning |
|---|---|
| `change.tif` | `uint8`; `1` changed, `0` unchanged, `255` no data (nodata in either image). Tags record `method` and `threshold` |
| `threshold` | The magnitude above which a pixel counts as changed, in the rasters' own units |
| `change_fraction` | Share of **valid** pixels that changed |
| `valid_fraction` | Share of pixels valid in both images |

The two rasters must have the same shape and grid. On a scene that did not change, Otsu still splits the noise into two
groups and reports a change; pass `--min-magnitude` (an absolute floor in the raster's units) to prevent that.

### Prepare the pair first

Most bad change maps come from the preparation, not the method.

| Check | Why |
|---|---|
| Same grid, same size | The command stops with `before/after rasters differ in shape` otherwise |
| Same season and similar sun angle | Otherwise vegetation, shadows and snow look like change |
| Same radiometry (both surface reflectance, or both the same digital numbers) | A different scale makes everything "change" |
| Cloud and shadow masked in both | Cloud on one date is the biggest source of false change |

See [Preprocessing](../data/preprocessing.md) and [Radiometry](../data/radiometry.md).

## With a trained network

```bash
pygeovision ai change --before a.tif --after b.tif -o change.tif --method bit --backbone resnet50 --checkpoint bit.pth
```

| Method | What it is | Notes |
|---|---|---|
| `bit` | Bitemporal Image Transformer. Encoder `resnet18` (CLI default) or `resnet50` | Registry: `bit-r50` is the ResNet-50 variant. Run it with `--backbone resnet50` |
| `dsamnet` | Deeply supervised attention network | Registry `dsamnet` is the `resnet18` variant |
| `changeformer` | A compact siamese cross-attention network (~0.8 M parameters). **Not** the published transformer ChangeFormer | One fixed architecture; no `--backbone` |

The checkpoint is the **inner network's** `state_dict` (see [Train one](train.md#save-the-inner-network)), and the encoder must
match what you trained.

An under-trained network gives a confident-looking map. In the end-to-end test on synthetic data a two-epoch network marked
99.997 % of the scene as changed. Always inspect the result and validate it.

## In Python

```python
import torch
from pygeovision.models import get_model

net = get_model("bit-r50", num_classes=2, in_channels=4, pretrained=False)
net.load_state_dict(torch.load("bit.pth", weights_only=True)); net.eval()

with torch.no_grad():
    scores = net(before_chip, after_chip)          # each (1, 4, H, W); result (1, 2, H, W)
    change = scores.argmax(dim=1)                  # 1 = changed
```

Run this chip by chip for a large scene, with overlap, the same way as [tiled inference](../inference/tiles.md); the
`ai change` command does this for you.

## Pipelines

When the change you want is a known kind, a pipeline does the whole job (finds both dates, masks cloud, reports areas):

| Change | Pipeline |
|---|---|
| Anything, two dates | [`change_detection`](../pipelines/catalog/change_detection.md) |
| Burn severity | [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md) |
| Forest loss | [`forest_change`](../pipelines/catalog/forest_change.md), [`deforestation`](../pipelines/catalog/deforestation.md) (your weights) |
| City growth | [`urban_expansion`](../pipelines/catalog/urban_expansion.md), [`urban_growth`](../pipelines/catalog/urban_growth.md) (your weights) |
| Flooding | [`flood_mapping`](../pipelines/catalog/flood_mapping.md), [`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md) |

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| Whole scene "changed" | Radiometry or season differs, or a network is under-trained | Match dates and scale; check training metrics |
| `size mismatch` with a checkpoint | Encoder or band count differs | `--backbone`; same `in_channels` as training |
| Change along tile edges | Mis-registered images | Co-register the pair; even a one-pixel shift creates edge change |
