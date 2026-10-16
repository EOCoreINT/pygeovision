# Data in and out

**Functions:** `obia.load_raster`, `obia.save_raster`, `obia.RasterData`, `obia.labels_to_gdf`, `obia.rasterize_geometries`,
`obia.from_labels`, `obia.samples_from_geometries`, `obia.valid_mask`, and the plot helpers in `pygeovision.obia.viz`

## Why

OBIA moves between three representations of the same thing, and most mistakes happen at the joins:

| Representation | Looks like | Used for |
|---|---|---|
| **Image** | `(bands, rows, cols)` float array + transform + CRS | Segmentation and features |
| **Label raster** | `(rows, cols)` int array, `0` = nothing, `1…n` = object | Everything that maps objects to pixels |
| **Vector** | One polygon per object, attributes as columns | GIS, reference data, deliverables |

This page covers the functions that convert between them, and what each one assumes.

## Reading an image: `load_raster`

```python
from pygeovision import obia

raster = obia.load_raster("scene.tif")                                   # all bands
raster = obia.load_raster("scene.tif", bands=[4, 3, 2])                  # 1-based band numbers
raster = obia.load_raster("scene.tif", band_names=["blue", "green", "red", "nir"])   # override names
print(raster.image.shape, raster.band_names, raster.crs, raster.pixel_area)
```

```text
(4, 160, 160) ['blue', 'green', 'red', 'nir'] EPSG:32633 4.0
```

`load_raster(path, bands=None, band_names=None)` returns a `RasterData`:

| Member | Meaning |
|---|---|
| `image` | `float32` array, `(bands, rows, cols)`, whatever the file's data type was |
| `band_names` | From `band_names=` if you give it; else the GeoTIFF's **band descriptions**; else `b1, b2, …` (per band: a missing description becomes `b<n>`) |
| `transform`, `crs` | Copied from the file |
| `nodata` | The file's nodata value, **stored but not used**. Nothing in `obia` masks it (see below) |
| `shape` | `(rows, cols)` |
| `pixel_area` | `|a·e|` of the transform, in CRS units² (m² only if the CRS is projected). `None` without a transform |

Raises `ValueError: N band names for M bands` when `band_names` does not match the bands you select.

**Band names matter.** `obia.spectral_layers` and the CLI look for bands called `red`, `green`, `nir`
([overview](index.md#conventions)). A file with no descriptions gives `['b1', 'b2', 'b3', 'b4']` and no NDVI.
Fix it once when writing the file (`save_raster(..., band_names=[...])`) or on the command line side with
`rasterio` (`set_band_description`).

### Anything that is already an array

Every function that takes an `image` accepts a path, a `RasterData`, or an array. To make the conversion explicit:

```python
from pygeovision.obia.io import as_raster_data
r = as_raster_data(arr, band_names=["red", "nir"], transform=tr, crs="EPSG:32633")
```

A 2-D array becomes one band; anything else but 3-D raises `ValueError`.

### No-data and NaN

Pixels that are NaN or infinite in any band, or equal to the raster's `nodata` in every band, are **no data**. Segmentation gives
them label 0 and builds no object over them; features and the scale tools leave them out
([Real imagery](real-data.md#no-data-pixels)). `obia.valid_mask(raster)` returns the mask. An image with no valid pixel raises
`ValueError: the image has no valid pixels`. Do not fill gaps with zeros: a filled patch becomes an object of its own.

### Rescaling: `rescale_bands`

```python
from pygeovision.obia.io import rescale_bands
img255 = rescale_bands(raster.image)                    # percentile 2–98 → 0–255, per band
```

This is what `segment` does for you by default (`rescale=True`) so that `scale` values transfer between images. A
constant band becomes 0 everywhere. It is not exported at the top level of `obia`.

## Writing a raster: `save_raster`

```python
obia.save_raster("labels.tif", seg.labels, raster.transform, raster.crs)                       # 2-D → 1 band
obia.save_raster("scene_copy.tif", raster.image, raster.transform, raster.crs, raster.band_names)  # 3-D → bands
```

`save_raster(path, array, transform=None, crs=None, band_names=None, dtype=None)` writes a GeoTIFF; a 2-D array becomes
one band. `dtype` defaults to the array's own type. Pass `band_names` to write band descriptions, which `load_raster`
reads back. Without a transform the file gets an identity transform (pixel coordinates) and no CRS.

Choose the type yourself when it matters: object ids belong in `int32` (the default if the array is `int32`; not `uint8`,
which stops at 255 objects), class codes in `uint8`.

## Objects to polygons: `labels_to_gdf`

```python
gdf = obia.labels_to_gdf(seg.labels, raster.transform, raster.crs)
gdf = obia.labels_to_gdf(seg.labels, raster.transform, raster.crs,
                         attributes=feats[["area_m2", "mean_ndvi"]])        # join columns by object id
gdf.to_file("objects.gpkg", driver="GPKG")
```

```text
   obj_id                                           geometry
0       1  POLYGON ((352000 5297000, 352000 5296872, 3520...
1       2  POLYGON ((352022 5297000, 352022 5296992, 3520...
['obj_id', 'geometry', 'area_m2', 'mean_ndvi']
```

- One row per object id, geometry type `Polygon` (a `MultiPolygon` only if an id is not connected, which segmentation
  prevents).
- Polygon corners follow pixel edges, so the shapes are stair-stepped. Simplify with `gdf.geometry.simplify(pixel_size/2)`
  for display; keep the exact version for area calculations.
- Without a transform the coordinates are pixel units with the origin at the top-left corner of the raster and **y growing
  with the row index**, so the picture is upside-down in a GIS. Pass the transform.
- `attributes` is any DataFrame indexed by object id. `id_column` (default `obj_id`) is the join key.
- `SegmentationResult.to_vector()` and `ClassificationResult.to_vector(labels, transform, crs, features=None)` call this
  for you; the classification variant adds `class`, `stability` and `score_<class>`.

## Reference polygons to a raster: `rasterize_geometries`

```python
ras, codes = obia.rasterize_geometries(ref_gdf, raster.shape, raster.transform)                # one id per feature
ras, codes = obia.rasterize_geometries(ref_gdf, raster.shape, raster.transform, "class")      # one code per class value
```

```text
int32 28 [(0, 1), (1, 2), (2, 3)]                          # first call: max id 28; codes = {row position: id}
{'cropland': 1, 'bare_field': 2, 'forest': 3, 'water': 4, 'road': 5, 'building': 6}      # second call
```

| Call | Raster contains | `codes` |
|---|---|---|
| Without `value_column` | A distinct id `1…N` per feature, `0` elsewhere. This is the `ref_labels` that [`segmentation_quality`](accuracy.md#segmentation-quality) and `scale_vs_reference` expect | `{row position: id}` |
| With `value_column` | The class code `1…K` (in order of first appearance) | `{value: code}` |

Behaviours to know:

- Background is `0`; overlapping features: **later rows overwrite earlier ones**.
- `all_touched=False`: a pixel belongs to a polygon only if its centre is inside, so polygons smaller than a pixel vanish.
- Empty or missing geometries are skipped. An empty GeoDataFrame gives an all-zero raster.
- Reproject first: `ref_gdf.to_crs(raster.crs)`. A reference in another CRS lands outside the raster and the result is
  all zeros (which `segmentation_quality` then rejects).
- Class codes follow *order of appearance*, not alphabet. If you need a fixed mapping, build it yourself (see the
  [`pixel_report` warning](accuracy.md#pixel-accuracy-pixel_report)).

## Reference polygons to object classes: `samples_from_geometries`

```python
samples = obia.samples_from_geometries(seg.labels, ref_gdf, "class", raster.transform, min_coverage=None)
```

Returns a Series (class by object id) in which each object takes the class of the reference geometry covering most of it.
It is the bridge from digitised reference data to training samples and to accuracy assessment. Details, the point
behaviour and `min_coverage` are in [Classification → Getting samples](classification.md#getting-samples). Pass `crs=raster.crs` when the
reference is in another coordinate system. Land-cover *maps* as training data, cloud holes and downloading a scene are in
[Real imagery](real-data.md).

## Objects from somewhere else: `from_labels`

```python
seg = obia.from_labels(mask_array, image="aerial.tif")
seg = obia.from_labels(mask_array, image=raster)           # or leave image out for a result without the image attached
```

`from_labels(labels, image=None, *, band_names=None, transform=None, crs=None, method="external")` wraps any 2-D integer
raster as a `SegmentationResult`. `method` is just a label recorded in the result. With a bare array as `image`, pass
`transform` and `crs` to keep georeferencing; `labels` that are not 2-D raise `ValueError: labels must be 2-D`. Use it for SAM masks
([SAM labeling](../labeling/sam.md)), objects exported from other software, rasterised cadastral parcels, or hand-drawn polygons
burnt in with `rasterize_geometries`.

- Ids that are not connected are split; ids are renumbered `1…n`. A test with labels 7 (two adjacent blocks) and 3
  gave 2 objects, `[1, 2]`.
- `0` stays background and is never an object: if your raster uses `0` for a real region, add 1 to every id first.
- `ValueError: image (160, 160) and labels (10, 10) differ in shape` when the grids disagree. Resample first.
- Because the object ids come from outside, `seg.params` is empty.

## Looking at results: `pygeovision.obia.viz`

The plot helpers import matplotlib only when called, so the rest of the package works without it. Each returns a
`Figure`.

```python
from pygeovision.obia.viz import show_overlay, show_classes, plot_scale

fig = show_overlay(seg, bands=(2, 1, 0), title="scale 20")               # image with yellow object outlines
fig = show_classes(result, seg.labels, raster=raster, bands=(2, 1, 0))   # image next to the classified objects
fig = plot_scale(scale_table)                                            # a table from scale_diagnostics / scale_vs_reference
fig.savefig("figure.png", dpi=150)
```

| Function | Shows | Notes |
|---|---|---|
| `show_overlay(seg, bands=(0, 1, 2), title=None, ax=None)` | The image stretched to 0–255 with outlines | `bands` are *indices* (not 1-based). `(2, 1, 0)` gives red-green-blue for a blue,green,red,nir file. Needs the image attached to `seg` |
| `show_classes(result, labels, raster=None, bands=(0, 1, 2), colors=None, title=None, ax=None)` | Objects coloured by class, with a legend. With `raster` it draws the image alongside | `colors` is `{class: "#hex"}`. Unknown classes get a default palette. Class `0` (unclassified) is white |
| `plot_scale(df, ax=None)` | For `scale_vs_reference`: over, under and `D` against scale, the best marked. For `scale_diagnostics`: ROC-LV with candidates starred | |

`seg.overlay(bands, color)` returns the outline picture as a plain `uint8` array if you want to save it yourself.

## The engine: `RegionMerger`

Advanced. `obia.RegionMerger(image, init_labels, color_weight=0.9, compactness_weight=0.5, band_weights=None)` is the
merging engine behind `multires`, `segment_levels` and the scale tools. `merger.run(scale)` merges until the cheapest
remaining merge costs `scale²`; because merging only ever increases, you can call `run` with a series of rising scales,
taking `merger.labels()` after each. The input must be a `(bands, rows, cols)` array (ideally stretched to 0–255
already) and an integer starting segmentation. `run` raises `ValueError` for a negative scale, and the constructor for
weights outside `[0, 1]`. Most people never need it; use `segment`, `segment_levels` or the scale functions.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `['b1', 'b2', …]` instead of band names; no NDVI | The file has no band descriptions | `band_names=` when loading, or write descriptions |
| `ValueError: N band names for M bands` | Wrong number of names | One name per band you load |
| `ValueError: unmasked NaN values in image are not supported` | Gaps in the image | Fill them and plan for a "nodata" object |
| Polygons appear mirrored/upside-down in QGIS | Saved without transform (pixel coordinates) | Pass `transform` and `crs` |
| `area_m2` tiny or `NaN` | Geographic CRS or no transform | Reproject to a metric CRS; pass the transform |
| `rasterize_geometries` returns zeros | Reference in a different CRS, or outside the scene | `to_crs`, check bounds |
| Reference polygon missing from the raster | Smaller than a pixel centre | Use bigger polygons, or points via `samples_from_geometries` |
| Object ids in a GeoTIFF stop at 255 | Saved as `uint8` | Save labels as `int32` |
| `from_labels` produces more objects than ids | Non-contiguous ids were split | Expected: every object is one connected piece |
| `ModuleNotFoundError: matplotlib` | Plot helpers need it | `pip install matplotlib` |
| `ValueError: no image attached to this result` | `overlay` on a result built without an image | Pass `image=` to `from_labels` or `segment` |
| Figure shows odd colours | Wrong `bands` indices | They are zero-based indices into the file's band order |

```{seealso}
[Segmentation](segmentation.md) · [Classification](classification.md) · [Accuracy](accuracy.md)
```
