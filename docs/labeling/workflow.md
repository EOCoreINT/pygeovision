# Worked example: from a bounding box to training chips

This page strings the labeling tools together for one realistic job: **a building-segmentation training set for a
scene**, labels from OSM and Microsoft footprints, checked and cleaned. Every command works the same on your own
area; replace the paths and the box.

```{note}
The imagery download step uses the data layer ([Search and download](../data/search-and-download.md)). Network steps need
internet; the numbers in the "you should see" blocks are the *shape* of what to expect, not results for any real place.
```

## 0. What you need

```text
data/scene.tif       your imagery (any CRS), e.g. a Sentinel-2 or aerial GeoTIFF
BBOX                 the scene's footprint in degrees: (lon_min, lat_min, lon_max, lat_max)
```

Get the box from the file itself, in degrees, so you never mix up the order:

```python
import rasterio
from rasterio.warp import transform_bounds

with rasterio.open("data/scene.tif") as src:
    BBOX = transform_bounds(src.crs, "EPSG:4326", *src.bounds)      # (lon_min, lat_min, lon_max, lat_max)
print(BBOX)
```

## 1. Label from each source, on the imagery's grid

```python
from pygeovision.labeling import OSMLabeler, MicrosoftBuildingsLabeler

osm = OSMLabeler().label(BBOX, ["buildings"], "labels/osm.tif", reference_raster="data/scene.tif")
ms = MicrosoftBuildingsLabeler().label(BBOX, "labels/ms.tif", reference_raster="data/scene.tif", all_touched=True)
for name, r in (("osm", osm), ("microsoft", ms)):
    print(name, r["success"], r.get("n_features", r.get("n_buildings")), r["error"])
```

You should see a count for each, and `error` as `None`. If one says `success False`, read its `error` and go to the page for
that source. You can carry on with one source.

## 2. Look at them (do not skip this)

Open in QGIS: `data/scene.tif`, then `labels/osm.tif` and `labels/ms.tif` on top with 50% transparency, plus
`labels/osm_vectors.geojson` as an outline. Check three or four places:

- Are buildings **where the imagery shows them**, or shifted by a few pixels?
- Are there big areas with buildings in the imagery and none in the label? (Unmapped area.)
- Are there labels over bare ground? (Demolished, wrong date.)

If the labels are shifted consistently, the *imagery* may be mis-registered; fix that first.

## 3. Combine and score

```python
from pygeovision.labeling import AutoLabelPipeline, LabelQualityAssessor

result = AutoLabelPipeline(
    sources=["osm", "microsoft_buildings"], fusion="union", reference_raster="data/scene.tif",
).run(BBOX, output_dir="labels/fused/", categories=["buildings"])
print(result["sources_succeeded"], result["fused_path"])
print(result["quality"]["quality_grade"], result["quality"]["recommendations"])
```

Why `union` with two sources: with only two, a majority vote has no tie-break other than "first source wins". `union` says
"a building if either source says so" and you clean up afterwards. With a third source (Google, if it covers your area)
`majority_vote` is the better choice. See [Combining sources](pipeline.md).

## 4. Clean

```python
qa = LabelQualityAssessor()
cleaned = qa.clean(result["fused_path"], "labels/buildings_clean.tif", min_area_m2=15,
                   fill_holes=False, smooth=False)
print(cleaned["n_components_removed"], cleaned["before_label_pixels"], "->", cleaned["after_label_pixels"])
print(qa.assess("labels/buildings_clean.tif")["quality_grade"])
```

`fill_holes=False, smooth=False` keeps courtyards and the gaps between houses, which matter for building masks. The
returned counts show how much dust was dropped; if `after` is far below `before`, `min_area_m2` is too large for your
resolution.

## 5. Put the classes in the form the trainer wants

Building labels from these sources are already 0 (background) and 1 (building). Where the label also has no-data
(`255`) from a source that does not cover part of the scene, that is correct: the trainer skips it.

If you used a **multi-class** OSM raster (values 0, 1, 2, 4, 7 …), renumber to `0…N-1` before training:

```python
import numpy as np, rasterio

keep = [0, 1, 2, 4, 7]                                  # values that occur, in the order you want
lut = np.full(256, 255, np.uint8)                       # anything not listed becomes "ignore"
for new, old in enumerate(keep):
    lut[old] = new
with rasterio.open("labels/osm_multi.tif") as src:
    lab, prof = src.read(1), src.profile
prof.update(nodata=255)
with rasterio.open("labels/osm_multi_01234.tif", "w", **prof) as dst:
    dst.write(lut[lab], 1)
print({old: new for new, old in enumerate(keep)})       # keep this mapping with your model
```

Then `num_classes = len(keep)`. See [Preparing training data](../training/data.md#rules-for-labels).

## 6. Cut chips

```console
$ pygeovision ai chips --image data/scene.tif --label labels/buildings_clean.tif \
      --output data/chips --chip-size 256 --prefix scene1
✓ 140 chips exported → data/chips
```

Because every label was written with `reference_raster="data/scene.tif"`, the grids match and the command does not refuse.
Then [split by area](../training/data.md#splitting-train-and-validation) and train.

## 7. After the first model: find what to fix

Predict on a different part of the area, look at the errors, and choose what to label by hand next with
[active learning](active.md). Correct those chips (or the source labels), retrain, and compare the validation score.

```{admonition} Keep a hand-checked test set
:class: important
Take 30–50 chips and label them carefully by hand, outside OSM and the footprint datasets. Never train on them. This is
the only way to know how good the final model is: scoring a model on the same noisy labels it was trained on measures
how well it copied the noise.
```

## Variant: no map data at all

1. [Foundation clusters](foundation.md) or [SAM masks](sam.md) on a very-high-resolution image.
2. Hand-label a **small** area (a few hundred metres) in QGIS and save it as a class raster on the same grid.
3. [`assign_classes`](naming-segments.md) transfers those names to every segment in the scene.
4. Check the result, then continue from step 4 above.

## Variant: Global South, thin OSM

1. `GoogleBuildingsLabeler` and `MicrosoftBuildingsLabeler` on the same box.
2. Fuse with `majority_vote` and `categories=["buildings"]` for OSM as the third voice.
3. Treat pixels where only one source says "building" with caution; for a stricter set use `intersection`.

```{seealso}
[Labeling overview](index.md) · [Training](../training/index.md) · [Building footprints pipeline](../pipelines/index.md)
```
