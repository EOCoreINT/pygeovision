# Change detection

Change detection finds **where something changed between two dates**. The answer is a map: changed, unchanged, and no
data. A method that needs no training handles the common case; a trained siamese network learns which changes *matter*
(new buildings, not seasonal colour).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Models
:link: models
:link-type: doc
BIT, DSAMNet and the compact ChangeFormer, with measured sizes and shapes.
:::

:::{grid-item-card} Train one
:link: train
:link-type: doc
Stacking two dates, wrapping a siamese network for `GeoTrainer`, and testing it.
:::

:::{grid-item-card} Predict
:link: predict
:link-type: doc
`ai change` with or without a checkpoint, reading the outputs and what can go wrong.
:::
::::

## The workflow

```{figure} ../_static/img/flow-change-detection.svg
:alt: Change detection: align two dates, difference or network, threshold, change map
:width: 100%
:class: pgv-figure

Two aligned dates in, one change map out. The branch on the left needs no training; the one on the right needs your checkpoint.
```

## Three ways to find change

| You have | Use | Needs |
|---|---|---|
| Two images and no labels | **Change vector analysis** (`spectral_diff`): `pygeovision ai change --before a.tif --after b.tif` | Nothing. Shows everything that changed, including seasons and shadows |
| A pipeline that finds the imagery | [`change_detection`](../pipelines/catalog/change_detection.md) and the specialised change pipelines | A bounding box and two dates |
| Pairs of images with a change mask | **A trained network** (`bit`, `dsamnet`, `changeformer`): [Train one](train.md) | Labelled pairs and a checkpoint |

Specialised pipelines that are change detection for one purpose: [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md)
(dNBR), [`forest_change`](../pipelines/catalog/forest_change.md), [`urban_expansion`](../pipelines/catalog/urban_expansion.md),
[`landcover_change`](../pipelines/catalog/landcover_change.md) and the flood pipelines. Prefer them when they fit your
question: each knows which change matters.

## What goes in and what comes out

Two-image networks take `model(before, after)` and return `(batch, classes, H, W)`:

| Model | Inputs | Output |
|---|---|---|
| `bit-r50`, `dsamnet`, `changeformer-mit-b0` | two `(2, 4, 64, 64)` tensors | `(2, 2, 64, 64)` |

`changeformer-mit-b0` also accepts one tensor with the two images stacked on the channel axis, `[before | after]`, which is
the form `GeoTrainer` can feed ([how](train.md)). The two rasters must have the same shape and grid; bring them onto one
grid first (`pygeovision preprocess clip` to a common bounding box, then `resample` to the same pixel size).

## Reading a change map

| Output | Meaning |
|---|---|
| `change.tif` | `uint8`; `1` changed, `0` unchanged, `255` no data (nodata in either image). Tags record `method` and `threshold` |
| `threshold` | The magnitude above which a pixel counts as changed, in the rasters' own units |
| `change_fraction` | Share of **valid** pixels that changed |
| `valid_fraction` | Share of pixels valid in both images |

An automatic threshold always splits the pixels into two groups, even on a scene that did not change. Use the optional
`--min-magnitude` floor, and compare the result with what you know about the place.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `before/after rasters differ in shape` | Different extent or resolution | `preprocess clip` to the same box and `preprocess resample` to the same pixel size |
| Change map is mostly "changed" with `spectral_diff` | Different sun angle, season or sensor between dates | Use dates in the same season; correct radiometry first; set `--min-magnitude` |
| `size mismatch` with `--method bit` | Encoder differs from training | Add `--backbone resnet50` (or `resnet18`) to match |
| Clouds appear as change | Cloud differs between dates | Mask cloud before differencing ([Preprocessing](../data/preprocessing.md)) |

```{toctree}
:hidden:
:maxdepth: 1

models
train
predict
```
