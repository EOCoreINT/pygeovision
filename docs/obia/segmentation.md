# Segmentation: making image objects

**Functions:** `obia.segment`, `obia.segment_levels`, `obia.from_labels` · **CLI:** `pygeovision obia segment`

## Why

Pixels are a poor unit for describing land. A single pixel has no shape, no texture and no neighbours worth speaking of;
a field has all three. Segmentation groups neighbouring pixels that belong together into **objects**, and everything
after it (features, rules, classification, accuracy) works on those objects.

The quality of the objects limits everything downstream: if a building is merged with the lawn beside it, no classifier can
recover the building.

## How to use

```python
from pygeovision import obia

seg = obia.segment("scene.tif", "multires", scale=20)      # a path, a RasterData, or a (bands, rows, cols) array
print(seg.n_objects, seg.method, seg.params)
seg.save("labels.tif")                                      # int32 GeoTIFF of object ids
seg.to_vector().to_file("objects.gpkg")                     # one polygon per object
rgb = seg.overlay(bands=(2, 1, 0))                          # uint8 preview with outlines (red, green, blue by index)
```

```console
$ pygeovision obia segment scene.tif --scale 20 --output-labels labels.tif --output-vector objects.gpkg
  ✓ 45 objects (multires) → labels.tif
  ✓ Object polygons → objects.gpkg
```

### `segment(image, method="multires", *, band_names=None, transform=None, crs=None, rescale=True, **params)`

| Argument | Meaning |
|---|---|
| `image` | A file path, a `RasterData` (from `load_raster`), or an array `(bands, rows, cols)` / `(rows, cols)` |
| `method` | One of `multires`, `slic`, `felzenszwalb`, `quickshift`, `watershed`, `chessboard`. Anything else raises `ValueError` listing them |
| `band_names`, `transform`, `crs` | Only needed when `image` is a bare array and you want georeferenced output |
| `rescale` | `True` (default) stretches each band to 0–255 with 2–98 percentile clipping first. Keep it: it makes `scale` mean the same thing on different images |
| `**params` | Algorithm parameters (below). CLI: `--param key=value`, repeatable; numbers are parsed |

Returns a `SegmentationResult`:

| Member | Meaning |
|---|---|
| `labels` | `int32` array of object ids `1…n` |
| `n_objects`, `method`, `params` | Provenance. `params` shows the values actually used, defaults included |
| `raster` | The `RasterData` the segmentation ran on (image, band names, transform, CRS) |
| `to_vector(attributes=None)` | GeoDataFrame, one row per object, `obj_id` column, optional attribute table joined by id |
| `boundaries()` | Boolean mask of pixels on an object edge |
| `overlay(bands, color)` | RGB preview; needs the image attached |
| `save(path)` | GeoTIFF of ids |

Every id is **one connected piece**. If an algorithm returns one id for two separate patches they are split, and ids are
renumbered `1…n`.

## The six methods

| Method | Idea | Main parameters (defaults) | Use when |
|---|---|---|---|
| `multires` | Start from tiny objects and keep merging the neighbours that make the union least heterogeneous (Baatz & Schäpe criterion) until the cost passes `scale²` | `scale=30`, `color_weight=0.9`, `compactness_weight=0.5` | **Default.** Object size is controlled by one knob, and shape can be weighed in |
| `slic` | k-means in colour + position (superpixels) | `n_segments=500`, `compactness=0.1`, `sigma=1.0` | Fast, regular objects; a good first pass or a tiling for later merging |
| `felzenszwalb` | Graph-based merging | `scale=100`, `sigma=0.8`, `min_size=20` | Irregular natural boundaries; very fast |
| `quickshift` | Mode seeking in colour + position | `kernel_size=3`, `max_dist=6`, `ratio=0.5`, `sigma=0` | Textured scenes where you want blobs |
| `watershed` | Flood the gradient image from a grid of seeds | `spacing=10`, `compactness=0` | Objects separated by clear edges |
| `chessboard` | Square tiles | `size=16` | A **baseline**. Use it to see how much a real method improves |

Note that `scale` does **not** mean the same for `multires` and `felzenszwalb`. In `multires` it is a stopping
threshold on a heterogeneity cost (a merge happens while its cost is below `scale²`). In `felzenszwalb` it prefers larger
components. Do not copy a value from one to the other.

### What the results look like

All six methods on the synthetic [demo scene](demo.md) (160 × 160 pixels, 28 true objects), scored against the true
objects ([Accuracy](accuracy.md#segmentation-quality)). **Lower D is better.**

| Method and setting | Objects | Over-segmentation | Under-segmentation | D | Time |
|---|---:|---:|---:|---:|---:|
| `multires`, scale 20 | 45 | 0.079 | 0.095 | **0.109** | 0.65 s |
| `felzenszwalb`, scale 100 | 376 | 0.225 | 0.151 | 0.247 | 0.06 s |
| `watershed`, spacing 10 | 256 | 0.352 | 0.238 | 0.388 | 0.09 s |
| `slic`, 200 segments | 159 | 0.486 | 0.251 | 0.488 | 0.08 s |
| `quickshift`, max_dist 6 | 72 | 0.324 | 0.435 | 0.490 | 0.18 s |
| `chessboard`, size 16 | 100 | 0.601 | 0.403 | 0.607 | 0.02 s |

```{note}
These are one synthetic scene with one setting each. The scene has crisp, flat-coloured objects, which suits region
merging; real imagery is messier, and every other method has settings that would do better than the ones shown. Use the
table to see what the numbers *look like*, not to rank the algorithms.
```

## `multires` in detail

`multires` repeatedly merges the pair of neighbours whose union raises heterogeneity least, and stops when the cheapest
possible merge would cost `scale²` or more. Heterogeneity combines a **colour** part (how much the standard deviation of
the bands rises when two objects are merged) and a **shape** part (how much less compact or smooth the union is).

| Parameter | Default | Effect |
|---|---|---|
| `scale` | 30 | **The main knob.** Small = many small objects; large = few large ones. 10 → 176 objects, 20 → 45, 60 → 22 on the demo scene |
| `color_weight` | 0.9 | Share of colour in the criterion, 0–1. `1.0` = colour only (objects can be ragged); lower values favour clean shapes |
| `compactness_weight` | 0.5 | Within the shape part: 1 = compact blobs, 0 = smooth edges |
| `band_weights` | equal | One weight per band. Raise the NIR weight to separate vegetation, for instance |
| `init` | `"auto"` | Starting objects: `"pixels"` (textbook, slow), `"superpixels"` (SLIC first, fast), or your own label array. `auto` picks pixels up to 10,000 pixels and superpixels above |
| `init_segments` | `None` | Number of starting superpixels (default about one per 16 pixels) |

CLI: `--scale`, `--color-weight`, `--compactness-weight`; others through `--param init=pixels`.

Starting from superpixels (the default for anything larger than ~100 × 100) is a speed trade-off: no object can be smaller
than a superpixel, so very small `scale` values do not give you single-pixel objects there. Use `init="pixels"` if you need
that and the image is small.

**Different from eCognition.** eCognition grows objects in passes using local mutual best fitting; here the globally cheapest
merge is always done next. Scale values have the same *meaning* as in textbooks because the bands are stretched to 0–255, but
the boundaries will differ.

### Speed and size

The merging is pure Python. Timings on the demo scene (4 bands, scale 20, default settings):

| Image | Time | Objects |
|---|---:|---:|
| 160 × 160 | 0.7 s | 45 |
| 320 × 320 | 2.7 s | 92 |
| 480 × 480 | 5.8 s | 166 |

Time grows roughly with the number of pixels. For scenes of tens of megapixels, cut tiles first, or use `slic`/`felzenszwalb`
(a fraction of a second at these sizes).

## Several scales at once: `segment_levels`

```python
levels = obia.segment_levels("scene.tif", scales=[10, 30])
fine, coarse = levels[10.0], levels[30.0]
```

One merging run is paused at each scale, so every coarse object is an **exact union** of fine objects. That true hierarchy is
what [super-object features](hierarchy-context.md) need. (Segmenting twice at different scales does not guarantee nesting.)
The result is a dict `{scale: SegmentationResult}` in ascending order. On the demo scene: scale 10 → 176 objects, scale 30 → 38.

## Using objects from elsewhere: `from_labels`

```python
seg = obia.from_labels(sam_mask_array, image="aerial.tif")      # SAM masks, an eCognition export, hand-drawn polygons
```

Wraps any integer raster as image objects. Pieces that share an id but are not connected are split; ids become `1…n`.
Mismatched image/label shapes raise `ValueError`. Combine with [SAM](../labeling/sam.md): SAM provides the outlines, OBIA the
features and classification.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| Far too many tiny objects | `scale` too small, or noisy image | Raise `scale`; smooth or despeckle SAR first |
| Objects swallow neighbours (a house merged with its lawn) | `scale` too large | Lower `scale`; raise `color_weight`; use the [scale tools](scale.md) |
| Ragged, noisy edges | Colour-only criterion | Lower `color_weight` (0.7–0.8) and raise `compactness_weight` |
| Square-ish, unnatural objects | Shape weighted too heavily, or SLIC grid | Raise `color_weight` |
| Small objects you expected are missing | Starting superpixels were too coarse | `init="pixels"` (small images) or raise `init_segments` |
| Very slow | Large image, pixel start | Use `init="superpixels"` (the auto default over 10,000 px), tile the image, or `slic` |
| `ValueError: unknown method` | Typo | Use one of the six names |
| `ValueError … band names for N bands` | `band_names` count differs from bands | Pass one name per band |
| Output georeferencing missing | Input was a bare array | Pass `transform=` and `crs=`, or load with `load_raster` |
| Different scales give different results on two images | Radiometry differs and `rescale=False` | Keep `rescale=True` |
| Objects ignore a band you care about | All bands weigh equally | `band_weights=[…]` |
| Results differ from eCognition | Different algorithm | Expected; see above |

```{seealso}
[Choosing a scale](scale.md) · [Object features](features.md) · [Accuracy assessment](accuracy.md)
```
