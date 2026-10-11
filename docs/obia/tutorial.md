# OBIA tutorial: from image to honest accuracy

This walk-through takes the [synthetic demo scene](demo.md) from raw bands to a classified, assessed map, and shows what each
step does, what number it produces and what to do when it goes wrong. Every output is real and reproducible: the script
at the end runs in under a minute on a laptop with no downloads.

You will:

1. make two scenes, one to build with and one to judge with
2. segment into objects and measure how good the objects are
3. choose the scale from evidence
4. describe the objects with features
5. classify with hand-written rules, and improve them from the confusion matrix
6. train a classifier from samples
7. assess honestly on the second scene
8. add context
9. export for a GIS
10. do the same from the command line

Requirements: `pip install pygeovision` with the OBIA dependencies ([overview](index.md#requirements)).

## Step 1: Make two scenes

```python
import pandas as pd
from pygeovision import obia

A = obia.make_demo_scene(seed=0)          # build everything here
B = obia.make_demo_scene(seed=1)          # judge everything here
print(A.raster.image.shape, len(A.reference))
```

```text
(4, 160, 160) 28
```

Scene A has 4 bands (blue, green, red, nir) of 160 × 160 pixels and 28 true objects. Scene B has the same kinds of
things in a different layout. We will **never look at B until step 7.** If you adjust anything after looking at B, B stops
being a test.

## Step 2: Segment, and measure the objects

```python
seg = obia.segment(A.raster, "multires", scale=20)
print(seg.n_objects)                                   # 45
quality, summary = obia.segmentation_quality(seg.labels, A.ref_labels)
print(round(summary["mean_over_seg"], 3), round(summary["mean_under_seg"], 3), round(summary["mean_D"], 3))
```

```text
45
0.079 0.095 0.109
```

45 objects for 28 true ones. The two error measures say how the objects fall short: 0.079 over-segmentation
(true objects are split a little) and 0.095 under-segmentation (some segments run over into neighbours). `D`, the single
combined score, is 0.109 (0 is perfect). The *median* D is 0.017, so most objects are almost exact and a few are poor. See
[Segmentation quality](accuracy.md#segmentation-quality).

In real work you do not have `A.ref_labels`; digitise 20–30 objects for this purpose, it is the best hour you can spend.

## Step 3: Choose the scale from evidence

```python
table = obia.scale_vs_reference(A.raster, [5, 10, 15, 20, 30, 40, 60, 80], A.ref_labels)
print(table.round(3))
print(table["mean_D"].idxmin())
```

```text
       n_objects  mean_over_seg  mean_under_seg  mean_D  mean_QR
scale
5.0          895          0.655           0.010   0.464    0.340
10.0         176          0.349           0.054   0.273    0.616
15.0           66          0.136           0.062   0.129    0.819
20.0          45          0.079           0.095   0.109    0.845
30.0          38          0.076           0.155   0.149    0.788
40.0          29          0.057           0.261   0.211    0.702
60.0          22          0.051           0.403   0.306    0.567
80.0          20          0.050           0.439   0.331    0.531
20.0
```

Small scales split things (895 objects, over-segmentation 0.655), large scales merge them (under-segmentation 0.439 at 80).
`D` is lowest at **20**, where the two errors cross. That is the scale we used. The reference-free
`scale_diagnostics` suggests 40 for the same scene, which is too coarse: use it only when you have no reference, and look at
the objects either way ([Choosing a scale](scale.md)).

## Step 4: Describe the objects

```python
layers = obia.spectral_layers(A.raster)                       # adds ndvi and ndwi
feats = obia.extract_features(seg.labels, A.raster, extra_layers=layers)
truth = obia.samples_from_geometries(seg.labels, A.reference, "class", A.raster.transform)
cols = ["area_px", "compactness", "elongation", "mean_ndvi", "std_nir", "brightness"]
print(feats[cols].groupby(truth).median().round(2))
```

```text
            area_px  compactness  elongation  mean_ndvi  std_nir  brightness
class
bare_field    158.0         0.33        2.03      -0.02     5.00      113.43
building       62.0         0.67        1.51      -0.08     6.25      120.50
cropland      172.0         0.35        1.68       0.60     5.40       82.40
forest       1559.0         0.53        1.40       0.69    13.91       54.53
road          309.5         0.15       19.82      -0.07     6.14       97.39
water        1443.0         0.46        2.23      -0.44     2.84       41.15
```

This table is your rulebook. `feats` has 45 rows and 52 columns; the class medians show where each class sits. Read it the
way a rule-writer would:

- **water** is the only class with negative NDVI and very low brightness.
- **forest** and **cropland** both have high NDVI (0.69 vs 0.60) but forest is far more variable (`std_nir` 13.9 vs 5.4).
- **building** and **bare_field** are spectrally alike (brightness 120 vs 113) but buildings are small and compact (62 px,
  0.67 vs 158 px, 0.33).
- **road** stands out by its elongation (19.8).

Using the ground truth to build this table is fair here, because it is scene A. On your own data you would build it from
your training samples.

## Step 5: Classify with rules

```python
rules = obia.RuleBasedClassifier([
    ("water",      "mean_ndwi > 0.3"),
    ("forest",     "mean_ndvi > 0.4 and std_nir > 8"),
    ("cropland",   "mean_ndvi > 0.55"),
    ("building",   "compactness > 0.55 and area_px < 400"),
    ("bare_field", "brightness > 90"),
])
result = rules.classify(feats)
report = obia.assess_classification(result, seg.labels, A.reference, "class", A.raster.transform)
print(round(report["overall_accuracy"], 3), round(report["kappa"], 3), report["n_objects"])
print(report["confusion"].astype(int))
```

```text
0.711 0.627 45
              bare_field  building  cropland  forest  road  unclassified  water
bare_field            13         0         0       0     0             0      0
building               2         8         0       0     0             0      0
cropland               0         0         8       8     0             1      0
forest                 0         0         0       2     0             0      0
road                   2         0         0       0     0             0      0
unclassified           0         0         0       0     0             0      0
water                  0         0         0       0     0             0      1
```

Rules go in order and the first match wins. 0.711 is a start, and **the confusion matrix is the to-do list**:

| Error | Count | Why (look at the objects) | Fix |
|---|---:|---|---|
| cropland called forest | 8 | They are tiny (20–117 px) slivers across field edges; mixed pixels inflate `std_nir` | Require `area_px > 300` for forest |
| road called bare field | 2 | There is no road rule | Add `elongation > 8` before the building and bare-field rules |
| building called bare field | 2 | Their `compactness` (0.28 and 0.50) is just under the 0.55 threshold | Leave (see below) |

Print the offending rows to be sure you are reading it correctly:

```python
wrong = feats.join(truth.rename("truth")).assign(predicted=result.classes)
wrong = wrong[wrong.truth != wrong.predicted]
print(wrong[["truth", "predicted", "area_px", "std_nir", "elongation"]].round(1).head(6))
```

```text
           truth     predicted  area_px  std_nir  elongation
obj_id
5       cropland        forest     25.0     17.7         1.0
10      cropland        forest     79.0     16.5         2.9
12      cropland        forest     24.0     15.2         1.3
16      building    bare_field     55.0     13.7         1.4
17      cropland        forest     27.0     29.0         1.7
18          road    bare_field    208.0      8.6        13.1
```

Apply the two fixes:

```python
rules2 = obia.RuleBasedClassifier([
    ("water",      "mean_ndwi > 0.3"),
    ("forest",     "mean_ndvi > 0.4 and std_nir > 8 and area_px > 300"),
    ("cropland",   "mean_ndvi > 0.55"),
    ("road",       "elongation > 8"),
    ("building",   "compactness > 0.55 and area_px < 400"),
    ("bare_field", "brightness > 90"),
])
```

Accuracy on A rises from 0.711 to **0.844** (kappa 0.793). And now the warning: **that number is not trustworthy.**
We read the errors on A and wrote rules that remove them; A is now part of the rule-writing. Step 7 tests whether the
improvement is real.

Why not fix everything? Chasing the last error means tuning to one object. The two buildings called bare field differ by a
few pixels from the buildings you got right; a rule for them would fit scene A and probably hurt elsewhere.

## Step 6: Learn from samples

```python
clf = obia.ObjectClassifier("rf", random_state=0).fit(feats, truth)
print(clf.feature_importances_.head(5).round(3))
```

```text
mean_nir      0.129
mean_ndwi     0.080
mean_ndvi     0.064
brightness    0.063
mean_blue     0.051
dtype: float64
```

A random forest trained on the 45 reference objects: 17 cropland, 13 bare field, 10 building, and only 2 forest, 2 road and 1
water. The feature importances show it leaning on near-infrared and the water and vegetation indices, which agrees with
the rulebook of step 4. (`feature_importances_` is an importance ranking over the features, not a measure of accuracy.)

Now the trap. Predict the same objects and assess them:

```python
trained = obia.assess_classification(clf.predict(feats), seg.labels, A.reference, "class", A.raster.transform)
print(trained["overall_accuracy"])          # 1.0
```

**Perfect, and meaningless**: the forest memorised those 45 objects. See [the training data trap](accuracy.md#the-training-data-trap).

## Step 7: Assess honestly

Scene B has different objects. Segment it *the same way* and extract the same features. Nothing is retuned.

```python
segB = obia.segment(B.raster, "multires", scale=20)
featsB = obia.extract_features(segB.labels, B.raster, extra_layers=obia.spectral_layers(B.raster))

def score(res):
    x = obia.assess_classification(res, segB.labels, B.reference, "class", B.raster.transform)
    w = obia.assess_classification(res, segB.labels, B.reference, "class", B.raster.transform,
                                   weights=featsB["area_px"])
    return x["overall_accuracy"], x["kappa"], w["overall_accuracy"]

print(score(rules.classify(featsB)))
print(score(rules2.classify(featsB)))
print(score(clf.predict(featsB)))
```

Results, 43 objects assessed:

| Method | Trained/tuned on | Accuracy on **A** | Accuracy on **B** | Kappa on B | Area-weighted on B |
|---|---|---:|---:|---:|---:|
| Rules, first version (5) | looked at nothing | 0.711 | **0.791** | 0.733 | 0.962 |
| Rules, after step 5 fixes (6) | A's errors | 0.844 | **0.860** | 0.817 | 0.987 |
| Random forest | A's 45 objects | 1.000 | **0.907** | 0.875 | 0.992 |
| Decision tree | A's 45 objects | | 0.791 | 0.720 | 0.692 |
| k-nearest neighbours | A's 45 objects | | 0.814 | 0.747 | 0.837 |
| Support vector machine | A's 45 objects | | 0.860 | 0.816 | 0.814 |

(`tree` and `forest` use `random_state=0`; `knn` and `svm` are untuned.)

What this tells you:

- **The honest number is the B column.** The forest's A figure (1.000) was memory; its B figure (0.907) is skill.
- **The rule improvement transferred**: 0.791 → 0.860 on B, a scene those fixes were not designed on. The slivers and the
  missing road rule are general problems, not quirks of one scene. That is the kind of fix worth keeping.
- **The random forest is best here**, but only by 0.047 over the improved rules: **two objects out of 43**. With so few
  objects, do not read much into the ranking among the top three. The decision tree's area-weighted figure (0.692) shows
  it made mistakes on large objects, which hurts more than count suggests.
- **Count and area disagree** by a lot for rules (0.791 vs 0.962): the errors are small objects. Report both.
- **Rare classes still fail.** The forest finds only 2 of 4 forest, 1 of 2 road and 1 of 2 water objects on B
  ([confusion matrix](classification.md#the-honest-comparison)). One training object per class cannot teach it.
  More samples of rare classes are worth more than a cleverer algorithm.
- **Kappa is lower than accuracy** in every row because it discounts agreement expected by chance.

```{important}
Even this is a single pair of scenes with 43 objects. Run it with other seeds and expect the order of the middle
methods to change. The robust conclusion is the *gap between A and B* for anything fitted to A.
```

## Step 8: Add context (optional)

Some errors are about neighbours rather than the object itself. [Hierarchy and context](hierarchy-context.md)
shows a second pass that gives leftover small objects the class they mostly touch (`class_context`), and what that is worth
on this scene: about one more correct object in 45. Add it when a specific mistake calls for it.

## Step 9: Export for a GIS

```python
final = clf.predict(featsB)
gdf = final.to_vector(segB.labels, B.raster.transform, B.raster.crs,
                      features=featsB[["area_m2", "mean_ndvi"]])
gdf.to_file("classified.gpkg", driver="GPKG")
obia.save_raster("classes.tif", final.to_raster(segB.labels), B.raster.transform, B.raster.crs)
print(final.summary(featsB["area_m2"]).round(3))
```

```text
            n_objects     area  area_share
class
bare_field         11  40228.0       0.393
building           13   3088.0       0.030
cropland           15  39200.0       0.383
forest              2  11824.0       0.115
road                1   2388.0       0.023
water               1   5672.0       0.055
```

`classified.gpkg` has one polygon per object with columns `obj_id`, `class`, `stability`, a `score_<class>` column per
class, and the features you attached. Open it in QGIS and colour by `class`, or by `stability` to see where the
classifier hesitated. `classes.tif` is a `uint8` raster with codes `1…K` in the order of `final.class_names` (alphabetical
here), `0` = unclassified. `area` is in square metres because the scene is in a projected CRS (UTM).

This map is **not validated by this export**; the numbers in step 7 are its accuracy statement. Quote them with the map,
and the date and scene they apply to.

## Step 10: The same on the command line

```console
$ pygeovision obia demo ./A --seed 0
$ pygeovision obia segment A/scene.tif --scale 20 --output-labels A_labels.tif
  ✓ 45 objects (multires) → A_labels.tif
$ pygeovision obia classify A/scene.tif A_labels.tif --rules rules.yaml --output A_rules.gpkg
$ pygeovision obia assess-classification A_labels.tif A_rules.gpkg A/reference.gpkg --count-objects
```

with `rules.yaml` as in [Classification](classification.md#rules-file-for-the-command-line). The command line trains on
samples and assesses against references in the same image. To hold data out, as in step 7, either keep the training and
validation geometries in two separate files, or use Python.

## The whole script

Everything above in one file ({download}`download <_examples/obia_tutorial.py>`). It prints the numbers quoted on this
page and writes `classified.gpkg` and `classes.tif` into the current folder.

```{literalinclude} _examples/obia_tutorial.py
:language: python
```

## What you learned

| Step | Lesson |
|---|---|
| 2–3 | Measure the segmentation before classifying; choose `scale` from evidence, not habit |
| 4 | A per-class feature table tells you which rules can work |
| 5 | The confusion matrix is a to-do list; look at the wrong objects, not only the counts |
| 6–7 | The same model scores 1.000 on its training objects and 0.907 on a new scene |
| 7 | Fixes that remove general problems transfer; fixes that chase single objects do not |
| 7 | Report count- and area-based scores, the confusion matrix, and how many objects |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| Numbers differ slightly from this page | Different package or library version | Small shifts are normal; large ones are not. Check `seg.n_objects` is 45 first |
| `ImportError` for scikit-image or similar | Optional dependency missing | `pip install scikit-image scikit-learn scipy geopandas rasterio` |
| `ValueError: cannot evaluate rule 'x'` | Column name typo, or unnamed bands | [Classification: what to do](classification.md#what-to-do-when) |
| Your rule scores differ between A and B | Overfit to A, or just a different layout | Compare with the table in step 7 |
| Segmentation is slower than expected | Pure Python merging | [Speed and size](segmentation.md#speed-and-size) |

```{seealso}
[Segmentation](segmentation.md) · [Choosing a scale](scale.md) · [Features](features.md) ·
[Classification](classification.md) · [Hierarchy and context](hierarchy-context.md) · [Accuracy](accuracy.md)
```
