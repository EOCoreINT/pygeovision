# Object-based image analysis (OBIA)

**Package:** `pygeovision.obia` · **CLI:** `pygeovision obia` (7 commands)

## Why OBIA

A neural network ([Models](../models/index.md), [Training](../training/index.md)) labels every **pixel**. That works well
when you have thousands of labelled chips and a GPU. Many jobs are not like that: you have a few dozen examples, you need
to **explain every decision** to a client or a regulator, you want field-sized polygons rather than pixel noise, or you
are teaching how classification works.

OBIA works on **objects** instead of pixels:

1. **Segment** the image into objects: groups of neighbouring pixels that look alike (a field, a roof, a pond).
2. **Describe** each object with numbers: mean colour, area, shape, texture, what it touches.
3. **Classify** the objects with rules you write, fuzzy descriptions, or a classifier trained on a handful of samples.
4. **Assess** the result honestly against reference data.

What you gain: small training sets, readable rules, polygons you can open in a GIS, and measures such as "this field is
elongated, bright and touches a road". What you give up: OBIA does not learn features from raw pixels, so a bad
segmentation or a poor choice of features caps the result, and it is slower and fiddlier than a trained network on very
large scenes.

## The pieces

| Step | What it does | Page | CLI |
|---|---|---|---|
| Segment | Six algorithms; the default merges pixels until objects are as large as `scale` allows | [Segmentation](segmentation.md) | `obia segment` |
| Choose the scale | Compare scales with or without reference objects | [Choosing a scale](scale.md) | `obia scale` |
| Describe | 50+ numbers per object: spectral, shape, texture, context | [Object features](features.md) | `obia features` |
| Classify | Rules, fuzzy sets, random forest/kNN/SVM/tree | [Classification](classification.md) | `obia classify` |
| Use hierarchy and neighbours | Coarse/fine levels, "next to a road", "contains buildings" | [Hierarchy and context](hierarchy-context.md) | (Python) |
| Assess | Segmentation quality (over/under), confusion matrix, kappa | [Accuracy assessment](accuracy.md) | `obia assess-segmentation`, `obia assess-classification` |
| Move data | Read/write rasters, objects to polygons, SAM masks as objects, plots | [Data in and out](data-io.md) | |
| Teaching scene | A synthetic scene with exact ground truth | [The demo scene](demo.md) | `obia demo` |

A complete walk-through with real output is the [OBIA tutorial](tutorial.md).

## Five minutes

```python
from pygeovision import obia

scene = obia.make_demo_scene()                                   # synthetic: bands blue, green, red, nir
seg = obia.segment(scene.raster, "multires", scale=20)           # 45 objects
feats = obia.extract_features(seg.labels, scene.raster, extra_layers=obia.spectral_layers(scene.raster))
rules = obia.RuleBasedClassifier([
    ("water", "mean_ndwi > 0.3"),
    ("cropland", "mean_ndvi > 0.55"),
    ("bare_field", "brightness > 90"),
])
result = rules.classify(feats)
print(result.summary(feats["area_m2"]))
report = obia.assess_classification(result, seg.labels, scene.reference, "class", scene.raster.transform)
print(report["overall_accuracy"], report["kappa"])
```

```text
              n_objects     area  area_share
class
bare_field           26  47208.0    0.461016
cropland             14  47868.0    0.467461
unclassified          4   1552.0    0.015156
water                 1   5772.0    0.056367
0.578 0.41          # overall accuracy, kappa (45 objects assessed)
```

Three rules cannot describe six classes, so forest, buildings and roads are wrong and the accuracy is poor (0.58). That is
the point of starting small: the confusion matrix tells you which rule to write next. The [tutorial](tutorial.md) goes
from here to 0.79 with rules and 0.91 with a trained classifier, and explains each number.

or the same on files from the command line:

```console
$ pygeovision obia demo ./obia_demo
$ pygeovision obia segment ./obia_demo/scene.tif --scale 20 --output-labels labels.tif
$ pygeovision obia features ./obia_demo/scene.tif labels.tif --output features.csv
$ pygeovision obia classify ./obia_demo/scene.tif labels.tif --rules rules.yaml --output classified.gpkg
$ pygeovision obia assess-classification labels.tif classified.gpkg ./obia_demo/reference.gpkg
```

## Requirements

`scikit-image`, `scikit-learn`, `scipy`, `pandas`, `geopandas`, `rasterio`, `shapely`. `matplotlib` for the plot helpers
(imported only when you call them). The tests skip cleanly when a dependency is missing.

## Conventions

| Item | Rule |
|---|---|
| **Image layout** | `(bands, rows, cols)` NumPy arrays, `float32` when loaded from a file |
| **Object ids** | Integers `1…n`, `0` is never an object. Every id is one connected piece (4-connectivity). Ids are the DataFrame index everywhere (`obj_id`) |
| **Band names** | Come from the GeoTIFF's band descriptions, else `b1, b2, …`. Names become feature-column names (`mean_nir`), lower-cased only if you wrote them that way, and punctuation becomes `_` |
| **Units** | `area_px` and `perimeter_px` are in pixels (perimeter in *pixel edges*). `area_m2` is in map units², i.e. square metres **only for a projected CRS**; in degrees it is square degrees |
| **Class codes in rasters** | `1…K` in the order of `result.class_names`, `0` = unclassified. The CLI prints the table |

```{important}
**Name your bands.** The command line reads band names from the GeoTIFF's band descriptions. A file with no descriptions
gets `b1, b2, b3, b4`, and then `spectral_layers` finds no `red`, `nir` or `green`, no NDVI/NDWI is made, and a rule such as
`mean_ndvi > 0.4` fails with "cannot evaluate rule". Add descriptions once:

    import rasterio
    with rasterio.open("scene.tif", "r+") as dst:
        for i, name in enumerate(["blue", "green", "red", "nir"], start=1):
            dst.set_band_description(i, name)
```

## What OBIA in PyGeoVision is, and is not

- The `multires` segmentation implements the **published Baatz & Schäpe criterion**. It is **not** eCognition's
  proprietary algorithm and will not reproduce eCognition's boundaries. Say so when you compare against eCognition output.
- Everything runs in Python on one machine, in memory. Scenes of a few thousand pixels on a side are comfortable; see
  [Segmentation](segmentation.md#speed-and-size) for timings.
- Rule strings are evaluated with `pandas.DataFrame.eval`. Treat a rules file as **code**, not as untrusted input.
- Accuracy is only as honest as your reference data. Read [Accuracy assessment](accuracy.md), especially the section on
  why scoring a classifier on its own training samples says nothing.

```{toctree}
:hidden:
:maxdepth: 1

tutorial
demo
segmentation
scale
features
classification
hierarchy-context
accuracy
data-io
```
