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
  ✓ <n> objects (multires) → labels.tif
  ✓ Object polygons → objects.gpkg
```

### `segment(image, method="multires", *, band_names=None, transform=None, crs=None, rescale=True, **params)`

| Argument | Meaning |
|---|---|
| `image` | A file path, a `RasterData` (from `load_raster`), or an array `(bands, rows, cols)` / `(rows, cols)` |
| `method` | One of `multires`, `slic`, `felzenszwalb`, `quickshift`, `watershed`, `chessboard`, `quadtree`, `multi_threshold`, `contrast_split`, `spectral_difference`. Anything else raises `ValueError` listing them |
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

## The ten methods

| Method | Idea | Main parameters (defaults) | Use when |
|---|---|---|---|
| `multires` | Start from tiny objects and keep merging the neighbours that make the union least heterogeneous (Baatz & Schäpe criterion) until the cost passes `scale²` | `scale=30`, `color_weight=0.9`, `compactness_weight=0.5` | **Default.** Object size is controlled by one knob, and shape can be weighed in |
| `slic` | k-means in colour + position (superpixels) | `n_segments=500`, `compactness=0.1`, `sigma=1.0` | Fast, regular objects; a good first pass or a tiling for later merging |
| `felzenszwalb` | Graph-based merging | `scale=100`, `sigma=0.8`, `min_size=20` | Irregular natural boundaries; very fast |
| `quickshift` | Mode seeking in colour + position | `kernel_size=3`, `max_dist=6`, `ratio=0.5`, `sigma=0` | Textured scenes where you want blobs |
| `watershed` | Flood the gradient image from a grid of seeds | `spacing=10`, `compactness=0` | Objects separated by clear edges |
| `chessboard` | Square tiles | `size=16` | A **baseline**. Use it to see how much a real method improves |
| `quadtree` | Split a block into four until every band inside it has a standard deviation `<= scale` | `scale=10`, `min_size=1`, `max_size=None` | Scenes with large flat areas and small detailed ones; the result is a set of square blocks, so it is cheap and predictable but never follows a diagonal edge |
| `multi_threshold` | Cut one layer into classes (multi-Otsu, or your thresholds) and make every connected patch of a class an object | `layer="brightness"`, `n_classes=3` or `thresholds=[...]` | One layer separates what you want: water on NDWI, vegetation on NDVI, a dark shadow class |
| `contrast_split` | Repeatedly split objects into a dark and a bright part at the threshold with the largest edge contrast | `layer="brightness"`, `levels=2`, `min_contrast=0`, `steps=32`, `min_size=1` | Isolating bright or dark things (buildings, water, shadows) from a varied background |
| `spectral_difference` | Merge neighbours whose mean spectra differ by at most `max_difference` | `max_difference=10`, `metric="max"`, `strategy="best_first"` | A cleanup step after any other method, or when "same colour" is the only criterion you want |

Note that `scale` does **not** mean the same for `multires`, `felzenszwalb` and `quadtree`. In `multires` it is a stopping
threshold on a heterogeneity cost (a merge happens while its cost is below `scale²`). In `felzenszwalb` it prefers larger
components. In `quadtree` it is a standard deviation in band units (0–255 after the default stretch). Do not copy a value
from one method to another.

### Comparing methods on your own image

No method wins everywhere, and a table of scores from someone else's scene would only tell you about that scene. To compare
on yours, digitise 20-30 reference objects (polygons around things that should each be one object: a pit, a field, a roof) and
score each method against them with `obia.segmentation_quality` ([Accuracy](accuracy.md#segmentation-quality)); lower D is
better. `chessboard` is the baseline: if a method does not beat square tiles, its settings are wrong for the image.

```python
ref_labels, _ = obia.rasterize_geometries(reference_polygons.to_crs(rd.crs), seg.labels.shape, rd.transform)
for name, params in {"multires": dict(scale=16), "felzenszwalb": dict(scale=100), "slic": dict(n_segments=300),
                     "chessboard": dict(size=16)}.items():
    seg = obia.segment(rd, name, **params)
    table, summary = obia.segmentation_quality(seg.labels, ref_labels)
    print(name, seg.n_objects, round(summary["mean_D"], 3))
```

Without reference objects, judge by eye: overlay the outlines on the image (`seg.overlay`) and look at the places that matter.

## `multires` in detail

`multires` repeatedly merges the pair of neighbours whose union raises heterogeneity least, and stops when the cheapest
possible merge would cost `scale²` or more. Heterogeneity combines a **colour** part (how much the standard deviation of
the bands rises when two objects are merged) and a **shape** part (how much less compact or smooth the union is).

| Parameter | Default | Effect |
|---|---|---|
| `scale` | 30 | **The main knob.** Small = many small objects; large = few large ones. The object count falls steeply as it rises; see the [scale diagnostics](scale.md) |
| `color_weight` | 0.9 | Share of colour in the criterion, 0–1. `1.0` = colour only (objects can be ragged); lower values favour clean shapes |
| `compactness_weight` | 0.5 | Within the shape part: 1 = compact blobs, 0 = smooth edges |
| `band_weights` | equal | One weight per band. Raise the NIR weight to separate vegetation, for instance |
| `init` | `"auto"` | Starting objects: `"pixels"` (textbook, slow), `"superpixels"` (SLIC first, fast), or your own label array. `auto` picks pixels up to 10,000 pixels and superpixels above |
| `init_segments` | `None` | Number of starting superpixels (default about one per 16 pixels) |
| `strategy` | `"best_first"` | Order of merging, see below |

CLI: `--scale`, `--color-weight`, `--compactness-weight`; others through `--param init=pixels`, `--param strategy=mutual_best`.

Starting from superpixels (the default for anything larger than ~100 × 100) is a speed trade-off: no object can be smaller
than a superpixel, so very small `scale` values do not give you single-pixel objects there. Use `init="pixels"` if you need
that and the image is small.

**Merge order (`strategy`).** `best_first` (default) always does the globally cheapest merge next. `mutual_best` is the
*local mutual best fitting* of Baatz & Schäpe (2000): in every pass two neighbours merge only if each is the other's
cheapest neighbour and the cost is below `scale²`, so objects grow evenly over the whole image instead of one region at a time.
Both stop in a state where every pair of touching objects costs at least `scale²`, but they can end in different partitions, and
`mutual_best` is typically 2–3 times slower in this implementation. Neither reproduces eCognition's boundaries exactly (it is a
proprietary implementation); scale values have the same *meaning* as in textbooks because the bands are stretched to 0–255.

### Speed and size

The merging is pure Python. Run time grows with the number of pixels and with how many objects survive, so a busy,
textured scene is slower than a flat one of the same size. Time a 250 × 250 crop first and scale up from there.

For scenes of tens of megapixels, cut tiles first, or use `slic`/`felzenszwalb`, which are several times faster.

## The new methods in detail

**`quadtree`.** Rows are split at `h // 2` and columns at `w // 2`, so any image size works. A block is homogeneous
only if *every* band has a population standard deviation of at most `scale` (give a band weight 0 in `band_weights` to leave it
out). `min_size` is the smallest block side, `max_size` forces splits of blocks larger than it. Neighbouring blocks are separate
objects even when they have the same colour; follow with `spectral_difference` to join them.

**`multi_threshold`.** `layer` is a band index (`0`), a band name (`"nir"`), `"brightness"` (mean of the bands), or `"ndvi"` /
`"ndwi"` (from bands named red/nir or green/nir, computed on the original values). Band and brightness layers are on the
stretched 0–255 scale unless you pass `rescale=False`. Give `thresholds=[t1, t2, ...]` (a pixel equal to a threshold goes to the
class above) or let multi-Otsu find `n_classes - 1` of them; the thresholds actually used are in `seg.params["thresholds"]`. CLI:
`--param thresholds=0.2,0.5`.

**`contrast_split`.** For each object, `steps` candidate thresholds are spread evenly between its lowest and highest value.
The edge contrast of a threshold is the mean absolute difference between neighbouring pixels (inside the object) that lie on
opposite sides of it; the threshold with the largest contrast is used, if each side keeps at least `min_size` pixels and the
contrast reaches `min_contrast`. Dark and bright parts become connected objects and the next level repeats. `levels` is the
maximum number of cuts along any chain; flat objects, or objects where no threshold qualifies, are left alone. This follows the
idea of eCognition's contrast split (separate dark from bright by the best edge) but is an independent, simplified
implementation.

**`spectral_difference`.** Starts from pixels (small images) or superpixels (`init`, as for `multires`) and merges the pair with
the smallest difference of mean spectra while it is `<= max_difference`. `metric="max"` uses the largest absolute difference over
the bands with weight above 0; `metric="mean"` the weighted mean. Means are recomputed from all pixels of a merged object. The
threshold is inclusive, unlike `multires`, where a cost equal to `scale²` does not merge.

## How these algorithms are checked

`tests/test_obia_segmentation_reference.py` compares each algorithm with something independent of the code under test: the
Baatz–Schäpe cost and merge order with a naive implementation that recomputes everything from pixel masks (several seeds,
weights and scales), `quadtree` with a recursive reference, the multi-Otsu thresholds with an exhaustive search over all threshold
pairs, `contrast_split` with a loop over pixel pairs, the wrappers of `felzenszwalb`, `slic` and `quickshift` with scikit-image
itself, and small cases (a 1×2 block, three stripes, a line of numbers) with values worked out by hand in the comments.

## Several scales at once: `segment_levels`

```python
levels = obia.segment_levels("scene.tif", scales=[10, 30])
fine, coarse = levels[10.0], levels[30.0]
```

One merging run is paused at each scale, so every coarse object is an **exact union** of fine objects. That true hierarchy is
what [super-object features](hierarchy-context.md) need. (Segmenting twice at different scales does not guarantee nesting.)
The result is a dict `{scale: SegmentationResult}` in ascending order.

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
| Cloud holes (NaN) in the image | Masked pixels | Nothing: they get label 0 and no object. See [Real imagery](real-data.md) |
| Output georeferencing missing | Input was a bare array | Pass `transform=` and `crs=`, or load with `load_raster` |
| Different scales give different results on two images | Radiometry differs and `rescale=False` | Keep `rescale=True` |
| Objects ignore a band you care about | All bands weigh equally | `band_weights=[…]` |
| Results differ from eCognition | Different algorithm | Expected; see above |

```{seealso}
[Choosing a scale](scale.md) · [Object features](features.md) · [Accuracy assessment](accuracy.md)
```
