# OBIA tutorial: from a real image to honest accuracy

This walk-through takes a **real Sentinel-2 image**, downloaded with PyGeoVision's own search and download tools, from raw
bands to a classified map, and says at each step what the step does, what to look at and what to do when it goes wrong.

It deliberately quotes **no result numbers**. They depend on your area, date and clouds, and a number from someone else's
scene would only be something to match. What the page gives you instead is the order of work, the checks that catch the usual
mistakes, and the points where a flattering score appears by accident.

You will:

1. get the data
2. segment into objects and look at them
3. choose the scale
4. describe the objects with features
5. classify with hand-written rules
6. train a classifier from weak labels
7. assess honestly
8. add context
9. export for a GIS
10. do the same from the command line

Requirements: `pip install pygeovision` with the OBIA dependencies ([overview](index.md#requirements)), a network, and the
credentials of the satellite provider you use (Planetary Computer and AWS Earth Search need none). The full script is at the end.

## Step 1: Get the data

```python
import pygeovision as pgv
from pygeovision import obia

client = pgv.PyGeoVision()
BBOX = (-1.70, 6.18, -1.655, 6.225)                    # lon/lat, about 5 km x 5 km around Obuasi, Ghana
rd = obia.fetch_scene(client, BBOX, "2024-01", "./download", bands=("blue", "green", "red", "nir"))
wc, _ = obia.fetch_worldcover(client, BBOX, "./download/worldcover.tif", like=rd)
valid = obia.valid_mask(rd)
print(rd.image.shape, f"{100 * (1 - valid.mean()):.1f}% no data")
```

`fetch_scene` uses the same search, download, cloud-mask and mosaic code as every pipeline. Pixels hidden by cloud or shadow
come back **NaN in every band**; OBIA treats them as no data. `fetch_worldcover` returns ESA WorldCover on the image's own grid.
Details and failure modes: [Real imagery](real-data.md).

Keep the area to a few kilometres. The region merging is pure Python: a 5 km window at 10 m (500 × 500 pixels) takes well under
a minute, a full Sentinel-2 tile does not finish in reasonable time. For larger areas tile the image or use `slic` / `felzenszwalb`.

What you will not have is a **reference**: points or polygons that someone digitised on high-resolution imagery. WorldCover is
a model's output, not the truth. Plan to make a reference of a few dozen points per class (an afternoon in QGIS over Google
or Maxar imagery); without it nothing below can be called an accuracy.

## Step 2: Segment, and look at the objects

```python
seg = obia.segment(rd, "multires", scale=16)
assert ((seg.labels == 0) == ~valid).all()             # label 0 is exactly the cloud/shadow pixels
print(seg.n_objects)
import matplotlib.pyplot as plt
plt.imshow(seg.overlay((2, 1, 0))); plt.show()          # red, green, blue with object outlines
```

Look at the picture. Objects should follow fields, pits, roofs, ponds. If one object covers a pit and the forest next to it,
the scale is too large; if a field is cut into dozens of pieces, too small. No score replaces this look
([Segmentation](segmentation.md)).

If you have digitised reference *objects* (20–30 polygons drawn around things that should each be one object), you can measure
the segmentation instead of judging it by eye: [Segmentation quality](accuracy.md#segmentation-quality).

## Step 3: Choose the scale

```python
print(obia.scale_diagnostics(rd, [8, 12, 16, 20, 30]).round(2))
```

`scale_diagnostics` needs no reference. It reports the object count and the local variance at each scale, and marks scales
where the *rate of change* peaks as candidates. It is a **heuristic**: it nominates scales to look at, it does not pick the
right one. Segment at two or three candidates and compare the overlays. With reference objects,
`obia.scale_vs_reference` measures over- and under-segmentation per scale and is the better guide ([Choosing a scale](scale.md)).

## Step 4: Describe the objects

```python
feats = obia.extract_features(seg.labels, rd, extra_layers=obia.spectral_layers(rd),
                              texture_layers=["nir"], transform=rd.transform)
CLASS_MAP = {10: "forest", 30: "farmland", 40: "farmland", 50: "built_up", 80: "water"}   # WorldCover codes
weak = obia.samples_from_raster(seg.labels, wc, CLASS_MAP, min_purity=0.7)["class"]
cols = ["area_px", "compactness", "elongation", "mean_ndvi", "mean_ndwi", "std_nir", "brightness"]
print(feats.loc[weak.index, cols].groupby(weak).median().round(2))
```

`feats` has one row per object and about fifty columns: spectral statistics per band and index, shape, texture and context.
The table groups the objects by their WorldCover class and shows the median of a few columns. This is your rulebook; read it
the way a rule-writer would:

- Which class is the only one with high NDWI? That is water; its threshold is a gap in the table.
- Which classes have similar NDVI but different `std_nir` or `compactness`? That is where shape and texture earn their keep.
- Which classes overlap on everything? No rule will separate them; you need more information or a different class scheme.

Two cautions. The classes are *WorldCover's*, so the table shows what WorldCover's labels look like in your image, errors
included. And objects whose pixels are mostly a class you left out of `CLASS_MAP` (shrub, wetland, ...) are not in the table at
all (`min_purity` leaves them out, see [Real imagery](real-data.md#weak-labels-from-a-map-samples_from_raster)).

## Step 5: Classify with rules

```python
rules = obia.RuleBasedClassifier([
    ("water",    "mean_ndwi > 0.2"),
    ("forest",   "mean_ndvi > 0.7"),
    ("farmland", "mean_ndvi > 0.45"),
    ("built_up", "compactness > 0.5"),
], default="unclassified")
rule_result = rules.classify(feats)
print(rule_result.classes.value_counts())
```

Rules run in order and the first match wins. The thresholds above are placeholders: **read yours off the Step 4 table**. Write
the most distinctive class first (water), the catch-all last, and use `default=` for what nothing matched. If objects land in
`unclassified`, that is information: look at them.

Rules are transparent, need no samples and can be explained to a regulator. Their thresholds are fitted to the scene you read
them from, so treat their score as an optimistic one until it is measured on data they were not tuned on.

## Step 6: Learn from weak labels

```python
ref = None                                             # your independent reference points, if you have them
import geopandas as gpd
# ref = gpd.read_file("reference_points.gpkg")         # a 'class' column; any CRS

ref_objs = (obia.samples_from_geometries(seg.labels, ref, "class", rd.transform, crs=rd.crs).index
            if ref is not None else [])
train = weak.drop(index=ref_objs, errors="ignore")      # objects you will test on are NOT training samples
clf = obia.ObjectClassifier("rf", random_state=0, class_weight="balanced").fit(feats, train)
result = clf.predict(feats)
print(clf.feature_importances_.head(5).round(3))
```

The forest learns from the WorldCover-labelled objects, so it also learns WorldCover's mistakes. `class_weight="balanced"` stops a
rare class (say, mine ground) from being drowned by a common one. `feature_importances_` shows what it leans on; it is a ranking,
not a measure of accuracy.

**The trap.** If you predict the training objects and score them, you will get a very high number. The forest has seen those
objects; the score measures memory. See [the training data trap](accuracy.md#the-training-data-trap). The line that drops
`ref_objs` from `train` is there so the objects you assess in Step 7 were never trained on.

## Step 7: Assess honestly

There are two different questions, and only one of them is accuracy.

**Agreement with the weak labels, on blocks the model did not train on.** Objects next to each other look alike, so a random
train/test split flatters the model. Hold out whole map blocks instead:

```python
from sklearn.model_selection import GroupKFold
import pandas as pd
H, W = seg.labels.shape
blocks = (feats["centroid_row"] // (H / 3)).astype(int) * 3 + (feats["centroid_col"] // (W / 3)).astype(int)
pooled = pd.Series(index=weak.index, dtype=object)
for tr, te in GroupKFold(3).split(weak.index, groups=blocks.loc[weak.index]):
    m = obia.ObjectClassifier("rf", random_state=0, class_weight="balanced").fit(feats, weak.iloc[tr])
    pooled.loc[weak.index[te]] = m.predict(feats).classes.loc[weak.index[te]]
held = obia.confusion_report(weak, pooled)
print(held["overall_accuracy"], held["n_objects"])
```

This says how well the model reproduces WorldCover in places it has not seen. It does **not** say how well it maps the ground:
where WorldCover is wrong, a correct prediction counts as a miss and a copied error counts as a hit. Expect it to vary a lot
between folds when one class (the mine, say) sits in one block.

**Accuracy: against an independent reference.**

```python
for name, res in {"rules": rule_result, "random forest": result}.items():
    a = obia.assess_classification(res, seg.labels, ref, "class", rd.transform, crs=rd.crs)
    print(name, a["overall_accuracy"], a["kappa"], a["n_objects"])
    print(a["per_class"].round(3))
```

Only objects a reference point falls in are assessed, so **always report `n_objects`**. Read the per-class table, not just the
overall figure: a classifier can be good overall and useless for the one class you care about. Report the confusion matrix too.
The rules' thresholds were read from this same scene, so their score is optimistic; the forest's `ref_objs` were held out of its
training, so its score is the fairer one.

```{important}
No reference points, no accuracy. The map is still useful for looking at, but quote it as a classification, not as having been validated.
```

## Step 8: Add context (optional)

Some errors are about neighbours rather than the object itself. [Hierarchy and context](hierarchy-context.md) shows a second
pass that gives leftover small objects the class they mostly touch (`class_context`). Add it when a specific mistake calls for it.

## Step 9: Export for a GIS

```python
gdf = result.to_vector(seg.labels, rd.transform, rd.crs, features=feats[["area_m2", "mean_ndvi"]])
gdf.to_file("classified.gpkg", driver="GPKG")
obia.save_raster("classes.tif", result.to_raster(seg.labels), rd.transform, rd.crs, dtype="uint8")
print(result.codes())                                  # raster code per class; 0 = no data / unclassified
print(result.summary(feats["area_m2"]).round(3))       # objects and area (m2) per class
```

`classified.gpkg` has one polygon per object with `obj_id`, `class`, `stability`, a `score_<class>` column per class and the
features you attached. Colour it by `stability` in QGIS to see where the classifier hesitated. Areas are in square metres
because the image is in a projected CRS. The cloud area is not in any object: report it next to the class areas.

Adding up mapped hectares is a *map summary*, not an area estimate. Classification errors do not cancel in a total, and small
classes suffer most. For a number you will publish, estimate area from an independent stratified sample with error adjustment
(Olofsson et al., 2014, *Remote Sensing of Environment* 148).

## Step 10: The same on the command line

```console
$ pygeovision obia fetch --bbox -1.70 6.18 -1.655 6.225 --date 2024-01 -o scene.tif --worldcover wc.tif
$ pygeovision obia segment scene.tif --scale 16 --output-labels labels.tif
$ pygeovision obia classify scene.tif labels.tif --weak-labels wc.tif \
      --class-map "10=forest,30=farmland,40=farmland,50=built_up,80=water" --output classified.gpkg
$ pygeovision obia assess-classification labels.tif classified.gpkg reference_points.gpkg --class-column class
```

`classify` prints how many objects it trained on and reminds you that the score needs an independent reference. The command line
assesses against the same objects it can see; to hold reference objects out of training as in Step 6, use Python.

## The whole script

Steps 1-9 in one file ({download}`download <_examples/obia_tutorial.py>`). It needs a network, and it only reports an accuracy
if you provide `reference_points.gpkg`.

```{literalinclude} _examples/obia_tutorial.py
:language: python
```

## What you learned

| Step | Lesson |
|---|---|
| 1 | Real imagery has holes: cloud pixels are no data, not objects |
| 2-3 | Look at the objects; the scale diagnostics only nominate candidates |
| 4 | A per-class feature table tells you which rules can work; weak labels carry their source's errors |
| 6 | Keep the objects you will test on out of training |
| 7 | Agreement with a map is not accuracy; accuracy needs an independent reference, and its `n_objects` |
| 9 | Mapped hectares are a summary, not an area estimate |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `RuntimeError: no usable Sentinel-2 scene` | No network, no credentials, or no clear scene for that month. The message ends with the reasons the download code logged | Read them; check the network; widen the date (`"2023-12/2024-02"`) or raise `cloud_cover_max` |
| The cell prints nothing for a long time | It does not: it prints a "still working, N s elapsed, X MB received" line every 10 s. If it really is silent, `verbose=False` was passed | Look for "nothing new for N s": that is a slow or stalled network, not a frozen kernel |
| `ImportError` for scikit-image or similar | Optional dependency missing | `pip install scikit-image scikit-learn scipy geopandas rasterio` |
| `ValueError: cannot evaluate rule 'x'` | Column name typo, or unnamed bands | [Classification: what to do](classification.md#what-to-do-when) |
| `ValueError: ... lie outside the image` | The reference is in another CRS or another place | Pass `crs=rd.crs`; check the coordinates |
| Segmentation is slower than expected | Pure Python merging on a large area | [Speed and size](segmentation.md#speed-and-size) |
| Very high accuracy | You scored training objects, or the reference came from the weak-label map | Step 6 and Step 7 |

```{seealso}
[Real imagery](real-data.md) · [Segmentation](segmentation.md) · [Choosing a scale](scale.md) · [Features](features.md) ·
[Classification](classification.md) · [Hierarchy and context](hierarchy-context.md) · [Accuracy](accuracy.md)
```
