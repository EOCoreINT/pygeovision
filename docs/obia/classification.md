# Classifying objects

**Classes:** `RuleBasedClassifier`, `FuzzyClassifier`, `ObjectClassifier` · **CLI:** `pygeovision obia classify`

## Why three kinds

How you classify depends on what you know.

| You have | Use | Strength | Cost |
|---|---|---|---|
| Expert knowledge, no samples ("water is dark in NIR; roads are long and thin") | **Rules** | Transparent: you can say why every object got its class | You must find thresholds yourself |
| Expert knowledge with gradual boundaries | **Fuzzy** | Soft thresholds and a per-object *stability* that flags the hard cases | More to set up |
| Labelled example objects | **Supervised** (random forest, kNN, SVM, decision tree) | Finds thresholds for you; the tree prints readable rules | Needs samples; can overfit; less explainable (except the tree) |

All three take the [feature table](features.md) and return a `ClassificationResult`.

```{note}
**Units.** Spectral statistics are in the image's own units. A scene from `obia.fetch_scene` is reflectance, 0 to 1, so
`std_nir` of a field is a few hundredths and `brightness` is about 0.1 to 0.4; the thresholds in the examples below are
written for that. NDVI and NDWI are unitless and transfer between scenes; raw band thresholds do not. Read every threshold
off your own feature table.
```

## `ClassificationResult`

| Member | Meaning |
|---|---|
| `classes` | `pd.Series` indexed by object id: class name per object |
| `class_names` | The classes, in a fixed order |
| `method` | `"rules"`, `"fuzzy-min"`, `"supervised-rf"`, … |
| `scores` | Per-class membership or probability (fuzzy, supervised); `None` for rules |
| `stability` | Best score minus second best (fuzzy, supervised). Near 0 = the classifier is torn between two classes |
| `codes()` | `{class: raster code}`; codes are `1…K` in `class_names` order, `0` is unclassified |
| `to_raster(labels)` | `uint8` raster with each object's code |
| `to_vector(labels, transform, crs, features=None)` | GeoDataFrame with `class`, `stability`, `score_<class>` and optional feature columns |
| `summary(area=None)` | Objects per class, plus area and area share if you pass the area column |

```python
result.summary(feats["area_m2"])      # DataFrame: n_objects, area, area_share per class
```

## Rule-based

```python
rules = obia.RuleBasedClassifier(
    [("water",      "mean_ndwi > 0.3"),
     ("forest",     "mean_ndvi > 0.4 and std_nir > 0.03"),
     ("cropland",   "mean_ndvi > 0.55"),
     ("road",       "elongation > 4 and area_px < 600"),
     ("building",   "compactness > 0.55 and area_px < 400"),
     ("bare_field", "brightness > 0.25")],
    default="unclassified",
)
result = rules.classify(feats)
```

**Rules are tried in order and the first match wins.** Put specific classes before general ones: an object that is both
"bright" and "compact" is a building only if the building rule comes first. Each condition is a pandas expression over the
feature columns; it must evaluate to True/False for every object.

Why does this matter? Because the first rule wins, an object can satisfy several conditions and only one decides.
`explain` shows exactly that:

```python
rules.explain(feats, obj_id=17)
#         class                                  rule  condition_true  decides
# 0       water                       mean_ndwi > 0.3           False    False
# 1      forest       mean_ndvi > 0.4 and std_nir > 0.03        False    False
# ...
# 5  bare_field                       brightness > 0.25          True     True
```

### Rules file for the command line

```yaml
# rules.yaml
default: unclassified
rules:
  - {class: water,      when: "mean_ndwi > 0.3"}
  - {class: forest,     when: "mean_ndvi > 0.4 and std_nir > 0.03"}
  - {class: cropland,   when: "mean_ndvi > 0.55"}
  - {class: building,   when: "compactness > 0.55 and area_px < 400"}
  - {class: bare_field, when: "brightness > 0.25"}
```

```console
$ pygeovision obia classify scene.tif labels.tif --rules rules.yaml --output classified.gpkg
```

It prints the number of objects classified and the objects per class, and writes the polygons with their class.

The CLI computes NDVI/NDWI itself (from bands named `red`/`nir`/`green`), so `mean_ndvi` is available in rules **if the
GeoTIFF's bands have descriptions** ([OBIA overview](index.md#conventions)).

```{warning}
Rules are evaluated with `pandas.DataFrame.eval`. Treat a rules file like code from the same person who runs it, not like
data from someone else.
```

### Writing rules that work

1. Start with the **easiest, most distinct** class (water: dark in NIR) and look at the result.
2. Add one rule at a time; after each, check the confusion matrix ([Accuracy](accuracy.md)) to see what is still wrong.
3. Use the [per-class medians](features.md#what-the-features-look-like-by-class) of your own data (group the feature
   table by the sample class) to choose thresholds, not guesses.
4. Prefer a feature that separates by a wide margin over one that is borderline.
5. Fix the threshold on one scene and *test on another*. A rule tuned on the scene you assess is overfit.

## Fuzzy

A rule says "NDVI above 0.55, yes or no". A fuzzy description says how well an object fits a class, from 0 to 1, per feature.

```python
from pygeovision.obia import FuzzyClassifier, LargerThan, SmallerThan, Trapezoid, Triangle, Gaussian

fz = FuzzyClassifier(
    {"water":    {"mean_ndwi": LargerThan(0.1, 0.4)},
     "forest":   {"std_nir": LargerThan(0.02, 0.05), "mean_ndvi": LargerThan(0.3, 0.5)},
     "cropland": {"mean_ndvi": LargerThan(0.4, 0.6), "std_nir": SmallerThan(0.02, 0.04)}},
    combine="min", min_membership=0.1,
)
result = fz.classify(feats)
print(result.stability.describe())
```

### Membership functions

| Function | Shape | Parameters |
|---|---|---|
| `LargerThan(a, b)` | 0 up to `a`, linear rise to 1 at `b`, 1 beyond. `a == b` is a hard threshold | |
| `SmallerThan(a, b)` | 1 up to `a`, linear fall to 0 at `b` | |
| `Trapezoid(a, b, c, d)` | 0 below `a`, rises to 1 at `b`, 1 until `c`, falls to 0 at `d` | needs `a ≤ b ≤ c ≤ d` |
| `Triangle(a, b, c)` | Peak 1 at `b` | |
| `Gaussian(mean, sigma)` | Bell curve | `sigma > 0` |

### Combining the features of one class

| `combine` | Meaning | Effect |
|---|---|---|
| `"min"` (default) | Fuzzy AND: every condition must hold | Strict. One weak feature sinks the class |
| `"product"` | Multiply | Even stricter |
| `"mean"` | Average | Compensating: a weak feature can be offset by strong ones |

An object takes the class with the highest membership; if even that is below `min_membership` it becomes `unclassified`. A
feature not in the table raises `KeyError` naming the class and listing available columns.

### Stability

`stability` = best membership minus second best. Near 1: the object clearly belongs. Near 0: two classes fit equally
well, or none does. Map it: the low-stability objects are where your description needs more work.

If many objects come out `unclassified` or with a low stability, the description is incomplete: look at those objects'
features and add the missing class or widen a range.

## Supervised

```python
samples = obia.samples_from_geometries(seg.labels, training_gdf, "class", rd.transform, crs=rd.crs)
# or, from a land-cover map: obia.samples_from_raster(seg.labels, wc, CLASS_MAP)["class"]
clf = obia.ObjectClassifier("rf").fit(feats, samples)
result = clf.predict(feats)
print(clf.feature_importances_.head())
```

### Getting samples

`samples_from_geometries(labels, gdf, class_column, transform=None, min_coverage=None, crs=None)` gives each image object the class
of the reference geometry that covers most of it:

- polygons: the dominant class must cover at least `min_coverage` (default 0.5) of the object, otherwise the object is
  left out (it straddles two classes and would teach the classifier noise);
- points: any hit counts (`min_coverage=0`);
- objects the geometries do not reach are not samples.

Returns a `pd.Series` of class names indexed by object id. To learn from an existing land-cover *map* instead of polygons, use
`samples_from_raster` ([Real imagery](real-data.md#weak-labels-from-a-map-samples_from_raster)). Use polygons **drawn inside** an object, or points, rather than
outlines you traced carelessly across boundaries.

### Methods

| `method` | Model | Notes |
|---|---|---|
| `"rf"` | Random forest (300 trees) | Strong default; gives `feature_importances_` |
| `"knn"` | k-nearest neighbours (k=3), standardised | The classic "nearest neighbour" of OBIA software |
| `"svm"` | Support-vector machine with probabilities, standardised | Good with few samples and many features |
| `"tree"` | Decision tree (max depth 4) | `export_rules()` prints readable if/else rules |

`ObjectClassifier(method, feature_columns=None, exclude=(centroid_row, centroid_col, centroid_x, centroid_y, parent_id), random_state=0, **model_params)`.
Extra keyword arguments go to the scikit-learn model (`n_estimators=500`, `max_depth=6`, …).

- **Location columns are excluded by default.** Otherwise the model can "learn the map" (objects in the top-left are crops)
  and look perfect on your scene and fail anywhere else. Pass `feature_columns=[...]` to choose features yourself.
- `NaN` and infinities are filled with each column's training median, so small objects without texture still classify.
- Fitting needs samples from **at least two classes** (`ValueError` otherwise).
- Class names are stored as strings.

A decision tree can be printed as rules:

```python
tree = obia.ObjectClassifier("tree", max_depth=3).fit(feats, samples)
print(tree.export_rules())
```

```text
|--- mean_nir <= <threshold>
|   |--- <feature> <= <threshold>
|   |   |--- class: <class>
...
```

Read like this, a tree is a rule set found by the data. Compare it with your own rules.

### Cross-validation (optimistic)

```python
clf.cross_validate(feats, samples, k=5)      # dict like confusion_report, plus "folds"
```

Stratified k-fold on the sample objects. Neighbouring objects look alike, so the score is **optimistic**; treat it as a
smoke test, not as accuracy. Every class needs at least 2 samples (`ValueError: every class needs at least 2 samples` if a
class has one; a rare class with a single sample object triggers it). `k` is lowered to the smallest class count.

## The honest comparison

Rules and a trained classifier answer different needs, and the comparison is only fair on data neither has seen:

1. Write the rules (or train the classifier) on one area, or on objects that are set aside from the assessment.
2. Assess both against the **same** independent reference points ([Accuracy](accuracy.md)), and report `n_objects`.
3. Look at the per-class table. A trained classifier typically does well on classes with many samples and badly on rare ones
   (**the signature of few samples**); rules do as well as their author's thresholds, and are optimistic on the scene they
   were read from.
4. Report both the object-count and the area-weighted figure; they answer "how many things are right" and "what share of
   the ground is right".

[The tutorial](tutorial.md#step-7-assess-honestly) does exactly this with rules and a weak-label random forest.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `ValueError: cannot evaluate rule 'x': name 'y' is not defined` followed by the available columns | A typo, or a column that does not exist (NDVI with unnamed bands) | Use a name from the list; name the bands |
| `ValueError: rule … must give True/False` | An expression returning numbers | Add a comparison (`mean_ndvi > 0.4`) |
| Many objects `unclassified` | Rules do not cover them | Add rules; look at those objects' features; or change `default` |
| A class is never assigned | An earlier rule takes everything | Use `rules.explain(...)` on one such object |
| Fuzzy: all `unclassified` | `min_membership` too high, or ranges miss your data | Print `fz.memberships(feats)`; adjust ranges using the per-class medians |
| `KeyError … uses feature … which is not in the feature table` | Wrong column name in a description | Check `feats.columns` |
| `ValueError: need samples from at least two classes` | Samples cover one class | Add samples of other classes |
| `no image object overlaps the sample geometries` (CLI) | CRS or extent mismatch, or polygons too small to cover half an object | Check overlap; lower `min_coverage`; use points |
| Supervised result is perfect | You assessed on your training samples | See [Accuracy](accuracy.md#the-training-data-trap) |
| Supervised result is poor on a new scene | Overfit to scene A, different radiometry | Train on several scenes; normalise the bands the same way; fewer, more robust features |
| Rare class never predicted | Too few samples | Add samples for it; random forest `class_weight="balanced"` through `**model_params` |
| `RuntimeError: call fit() first` | `predict` before `fit` | Fit first |
| `export_rules()` raises | Not a fitted tree | Use `method="tree"` |

```{seealso}
[Object features](features.md) · [Accuracy assessment](accuracy.md) · [Hierarchy and context](hierarchy-context.md)
```
