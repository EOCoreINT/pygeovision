# Accuracy assessment

**Functions:** `obia.assess_classification`, `obia.confusion_report`, `obia.pixel_report`, `obia.segmentation_quality`,
`obia.samples_from_geometries` · **CLI:** `pygeovision obia assess-classification`, `pygeovision obia assess-segmentation`

## Why

A map without an accuracy statement is an opinion. There are two separate questions, and PyGeoVision answers them with
separate tools:

| Question | Tool | Compares |
|---|---|---|
| Are the **objects** the right shape and size? | `segmentation_quality` | Segments vs. reference objects you digitised |
| Are the **classes** right? | `assess_classification`, `confusion_report` | Predicted class vs. reference class, per object |
| Is the **map** right, per pixel? | `pixel_report` | Class raster vs. reference raster |

A good segmentation does not guarantee good classes (the objects can be perfect and mislabelled), and good classes
do not guarantee good objects (a field cut in two can have both halves right).

```{important}
Every number on this page depends on the reference data. Reference that is small, convenient (only the easy objects) or
**the same data the classifier learned from** gives numbers that look good and mean little. Read
[The training data trap](#the-training-data-trap) before you quote an accuracy.
```

## Classification accuracy

### From reference polygons or points: `assess_classification`

```python
from pygeovision import obia

report = obia.assess_classification(
    result,                      # ClassificationResult from any classifier
    seg.labels,                  # the object-id raster that was classified
    reference_gdf,               # polygons or points with a class column
    "class",                     # that column's name
    rd.transform,
    weights=None,                # or a Series, e.g. feats["area_px"]
    crs=rd.crs,                  # reproject the reference to the image CRS
    min_coverage=None,           # see samples_from_geometries
)
print(report["overall_accuracy"], report["kappa"], report["n_objects"])
```

```console
$ pygeovision obia assess-classification labels.tif classified.gpkg reference_points.gpkg --class-column class --count-objects
  assessed <n> objects (unweighted)
  overall accuracy: <oa>   kappa: <k>

              producers_accuracy  users_accuracy     f1  reference_total  predicted_total
  <class>                  ...
  confusion (rows = reference, columns = predicted):
  ...
```

The command prints the counts, the overall figures, the per-class table and the confusion matrix. A class the rules never
predict has `users_accuracy` `NaN` (nothing was claimed) and a producer's accuracy of 0: the table tells you which rule is
missing.

How it works: each object gets the class of the reference geometry that covers most of it
([`samples_from_geometries`](classification.md#getting-samples)); objects the reference does not
reach are **not assessed**. That is why the report carries `n_objects`: always quote it next to the accuracy.

### Direct comparison: `confusion_report`

```python
rep = obia.confusion_report(reference, predicted, weights=None, labels=None)
```

`reference` and `predicted` are class names indexed by object id; only ids present in both are used. `weights` (a Series
indexed by id, for example `feats["area_px"]`) makes every cell add up *area* instead of counting objects.
`labels` fixes the order of the matrix rows/columns (default: sorted union).

The returned dict:

| Key | Meaning |
|---|---|
| `confusion` | DataFrame, **rows = reference, columns = predicted** |
| `overall_accuracy` | Correct ÷ total (sum of weights if weighted) |
| `kappa` | Agreement beyond chance. 1 = perfect, 0 = no better than guessing the marginals |
| `per_class` | DataFrame, one row per class, see below |
| `total`, `n_objects`, `weighted` | Counts (or total weight), objects assessed, whether weights were used |

`per_class` columns:

| Column | Also called | Meaning |
|---|---|---|
| `producers_accuracy` | recall; 1 − omission error | Of the objects that *are* this class, how many did the map find? |
| `users_accuracy` | precision; 1 − commission error | Of the objects the map *calls* this class, how many really are? |
| `f1` | | Harmonic mean of the two |
| `reference_total` | | Row sum |
| `predicted_total` | | Column sum |

A class with no reference objects (row sum 0) gets `NaN` producer's accuracy rather than zero.

### How to read the report

```text
overall accuracy <oa>   kappa <k>   (<n> objects)

confusion (rows = reference, columns = predicted)
per_class: producers_accuracy  users_accuracy  f1  reference_total  predicted_total
```

- **Rows tell you what was missed.** A class with 8 reference objects of which 5 were found has a producer's accuracy of
  0.63; the other three went to some other column of that row.
- **Columns tell you what was wrongly claimed.** A class that was predicted 13 times with 11 right has a user's accuracy of
  0.85; the two extras are in that column, in other rows.
- The **diagonal** is what was right. Perfect accuracy is a diagonal matrix.
- Rare classes drive the poor rows. With few reference objects a single mistake moves a class's accuracy by tens of points,
  so quote `reference_total` beside every per-class figure.

### Count or area? `weights`

Pass `weights=feats["area_px"]` and every cell adds up *area* instead of counting objects. Both are right; they answer
different questions.

| Weighting | Answers | Use when |
|---|---|---|
| Count (default in Python) | "What share of *objects* is right?" | Objects are the unit of interest (count the buildings) |
| Area (`weights=feats["area_px"]`; the CLI default `--weight-by-area`) | "What share of the *ground* is right?" | Area statistics matter (hectares of cropland) |

When the misses are small objects the area-weighted score is higher than the object count; if you report only the larger
number you hide how many objects were wrong. Report both, or the confusion matrix.

```{note}
Python's `confusion_report` / `assess_classification` default to **counting objects**; the CLI command
`obia assess-classification` defaults to **area weighting** (`--weight-by-area`; use `--count-objects` to switch). Compare
numbers from the two only when you have set them the same way.
```

### Pixel accuracy: `pixel_report`

```python
code = result.to_raster(seg.labels)            # codes 1..K in the order of result.class_names, 0 = unclassified
rep = obia.pixel_report(code, reference_codes, result.class_names)
```

`reference_codes` is a raster with codes `1..K` in the **same order as `class_names`**, and `0` where there is no
reference (those pixels are ignored). Unclassified pixels inside the reference count as errors. The dict has the same
keys as `confusion_report` plus `unclassified_pixels`.

Pixel and object-wise figures differ a little even on the same map, because object-wise assessment gives each object the
class of the reference polygon that covers most of it, whereas the pixel view counts every pixel.

```{warning}
**The class order must match.** `result.class_names` is alphabetical (`bare_field, building, cropland, …`). If your reference
raster numbers its classes in another order, feeding it to `pixel_report` as it is gives a wrong accuracy with no error
message: every number is "valid", just meaningless. Build the reference codes from `result.class_names`:

    lut = np.zeros(len(ref_names) + 1, int)
    for i, name in enumerate(ref_names, start=1):
        lut[i] = result.class_names.index(name) + 1
    reference_codes = lut[reference_raster]
```

## Segmentation quality

```python
table, summary = obia.segmentation_quality(seg.labels, ref_labels)
```

```console
$ pygeovision obia assess-segmentation labels.tif reference_objects.gpkg --output per_object.csv
```

It prints the number of reference objects and segments, the mean over- and under-segmentation, mean D and mean QR, and writes
the per-object table to the CSV.

`ref_labels` is an integer raster with one id per reference object and `0` elsewhere; make it from polygons with
[`rasterize_geometries`](data-io.md#reference-polygons-to-a-raster-rasterize_geometries). The CLI does that for you and reprojects
the reference to the raster's CRS.

For every reference object **R** the function finds the segment **S** that overlaps it most, then measures:

| Measure | Formula | Meaning |
|---|---|---|
| `over_seg` | 1 − \|R∩S\| ÷ \|R\| | 0: the reference object lies completely inside one segment. Higher: it is split across several |
| `under_seg` | 1 − \|R∩S\| ÷ \|S\| | 0: the best segment lies completely inside the reference. Higher: the segment spills into other things |
| `D` | √((over² + under²) ÷ 2) | Combined distance. **0 is perfect**; use it to compare settings |
| `QR` | \|R∩S\| ÷ \|R∪S\| | Jaccard overlap; 1 is perfect |
| `AFI` | (\|R\| − \|S\|) ÷ \|R\| | Area fit index. Positive: segment smaller than reference. Negative: larger |
| `n_parts` | count | Number of segments of which at least half lies inside the reference |

The table has one row per reference object (`best_segment`, the three areas, then these measures). The summary dict has
`n_reference`, `mean_over_seg`, `mean_under_seg`, `mean_D`, `median_D`, `mean_QR`, `n_segments_total`.

### Reading the per-object table

```text
        best_segment  ref_area_px  seg_area_px  overlap_px  over_seg  under_seg      D     QR    AFI  n_parts
ref_id
```

One row per reference object. Four patterns cover most rows:

- **D near 0** (over and under both small): the segment is the object.
- **High `over_seg`, `under_seg` 0**: *over-segmented*. The best segment lies entirely inside the reference but covers only
  part of it; several segments make up the object (`n_parts` > 1).
- **`over_seg` near 0, high `under_seg`** (AFI strongly negative): *under-segmented*. The segment holds the whole reference but
  is larger; it contains something else too.
- **Both high**: the worst case; the best segment only half-overlaps and is larger than the reference.

Sort by `D` to find the objects to look at: `table.sort_values("D", ascending=False).head()`. The mean can hide the spread
when most objects are very good and a few are poor: `mean_D` is then well above `median_D`. Look at both.

### Two kinds of error, two remedies

| Pattern | Means | Do this |
|---|---|---|
| High `over_seg`, low `under_seg` | Objects are split | Raise `scale` |
| Low `over_seg`, high `under_seg` | Objects are merged with things beside them | Lower `scale`; raise `color_weight` |
| Both high for the same object | Segmentation cannot separate this object spectrally | Different layers or `band_weights`, or different method |
| Both low everywhere | Segmentation matches the reference | Move on |

Use [`scale_vs_reference`](scale.md#with-reference-objects-scale_vs_reference) to scan many scales at once.

## The training data trap

The most common way to publish an impressive and meaningless accuracy is to score a classifier on the samples it learned
from. A supervised classifier can memorise its training objects; scoring those same objects only tests memory.

A random forest fitted to labelled objects reproduces those labels almost perfectly when it is asked about them again. The
score for those objects is then close to 1.0 whatever the quality of the labels or of the model; it only shows that the
forest remembers its training data. The score on objects it has never seen is lower, and for rare classes much lower. The gap
between the two is the cost of reporting the first number.

The command line walks straight into it, because `obia classify --samples reference.gpkg` trains on the reference and
`obia assess-classification` then checks against the same reference:

```console
$ pygeovision obia classify scene.tif labels.tif --samples reference.gpkg --class-column class --output classified.gpkg
  trained on <n> sample objects (<k> classes)
  note: judge accuracy with reference data that was NOT used for training
$ pygeovision obia assess-classification labels.tif classified.gpkg reference.gpkg
  overall accuracy: ~1.000 (it was trained on these objects)
```

The command prints the warning, and the near-perfect score is exactly what it warns about. For a held-out assessment, give the
CLI reference polygons that are **different** from the `--samples` file (digitise a validation set separately), or use Python
and drop the validation objects from the training set, as in the [tutorial](tutorial.md#step-6-learn-from-weak-labels).

With weak labels (`--weak-labels`, a land-cover map used as training data) the trap has a second form: the training labels
come from a map, and scoring against that map measures agreement with the map, not accuracy. `obia classify --weak-labels`
reminds you of this; assess against your own reference points.

Rules to keep your numbers honest:

1. **Hold data out.** Assess on objects that took no part in training or in tuning your rules: a different area, a
   different date, or at least spatially separate blocks.
2. **Separate your samples.** Split the reference into training and validation *before* you start. Neighbouring objects look
   alike, so a random split of objects from one image is still too optimistic. Split by region.
3. **Do not tune on the validation set.** Every time you adjust a threshold after looking at the validation score, the
   validation set turns into training data.
4. **Cross-validation is a smoke test.** `ObjectClassifier.cross_validate` reuses samples from one image, so it is optimistic.
5. **Rules are not exempt.** Thresholds read off the scene you then assess are fitted to that scene. The size-rule fix in
   [Hierarchy and context](hierarchy-context.md#a-worked-example-slivers-on-field-edges) improves the scene it was designed on;
   treat that improvement as an upper bound.
6. **Say how many.** Report `n_objects` and the confusion matrix, not just one number. With 40 objects and a class of 2,
   each object is 2.5 % of the total.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `ValueError: no object is covered by the reference geometries` | Reference does not overlap, wrong CRS, or polygons smaller than half an object | Check extent and CRS; try `min_coverage` lower, or use points |
| `ValueError: reference and predicted share no object ids` | Different segmentations, or predictions indexed differently | Assess the classification on the same label raster it was made on |
| `ValueError: weights are missing for some assessed objects` | The weight Series misses ids | Use `feats["area_px"]`, indexed by `obj_id` |
| `ValueError: reference raster contains no reference objects` | Reference did not rasterise (CRS/extent), or all background | CLI: check the CRS; Python: reproject first |
| `ValueError: segmentation and reference rasters differ in shape` | Different grids | Rasterise the reference onto the segmentation's grid |
| Accuracy is exactly 1.000 | You are assessing training samples | See [the trap](#the-training-data-trap) |
| Pixel accuracy is far below object accuracy | Class order mismatch in `pixel_report`, or objects unclassified | Rebuild the reference codes in the `class_names` order; check `unclassified_pixels` |
| Overall accuracy high but a class is wrong | Large classes dominate the average | Read `per_class`; report producer's and user's accuracy |
| Kappa is `NaN` | Chance agreement is 100 % (one class in both reference and prediction) yet some objects disagree | Report accuracy and the confusion matrix instead |
| `users_accuracy` is `NaN` | The class was never predicted, so there is nothing to be "user-correct" about | Read it together with a producer's accuracy of 0: the class is missing from your map |
| Area-weighted and count-based scores disagree strongly | Errors are in small objects (or large ones) | Report both; see [Count or area?](#count-or-area-weights) |
| CLI and Python give different numbers | CLI defaults to area weighting, Python to counting | Set `--count-objects` or pass `weights=` |
| Reference object class `NaN` | Reference has no class for that polygon | `dropna` the reference first; `confusion_report` already drops missing reference classes |
| `n_parts` is 0 | No segment lies at least half inside that reference object | The reference object is split into many small segments, or the segments are far larger than it |

```{seealso}
[Classification](classification.md) · [Choosing a scale](scale.md) · [Tutorial](tutorial.md)
```
