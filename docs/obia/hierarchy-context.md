# Hierarchy and context

**Functions:** `obia.segment_levels`, `obia.parent_map`, `obia.super_object_features`, `obia.sub_object_fraction`,
`obia.class_context` · **CLI:** none (Python only)

## Why

A feature table built from one segmentation describes each object *on its own*. People rarely classify that way. They say
"that small bright rectangle **beside a road** is a building", "that patch is part of a **larger** forest", "this field
is **almost surrounded by water**". Two kinds of extra information make those statements computable:

| Kind | Question it answers | Needs |
|---|---|---|
| **Hierarchy** (vertical) | What is the bigger object I belong to? What am I made of? | Two segmentation levels |
| **Context** (horizontal) | What classes are my neighbours? | A first classification |

Neither is a magic fix. Both are only worth adding when the plain features leave a specific, nameable mistake. The worked
example below shows one such mistake, the fix, and what the fix is really worth.

## Hierarchy

### Build a true hierarchy

```python
from pygeovision import obia

levels = obia.segment_levels(scene.raster, scales=[10, 30])   # one merging run, paused twice
fine, coarse = levels[10.0], levels[30.0]
print(fine.n_objects, coarse.n_objects)                        # 176 38
```

Because both levels come from the same run, **every coarse object is an exact union of fine objects** (checked on the demo
scene: every one of the 176 fine objects lies inside exactly one coarse object). Two independent `segment` calls at
different scales do not guarantee that. See [Segmentation](segmentation.md#several-scales-at-once-segment_levels).

### Which coarse object does a fine object belong to? `parent_map`

```python
pm = obia.parent_map(fine.labels, coarse.labels)
pm.head(5)
```

```text
obj_id
1    1
2    2
3    3
4    3
5    3
Name: coarse, dtype: int64
```

`parent_map(fine_labels, coarse_labels)` returns a Series indexed by fine id whose value is the coarse object that covers
**most** of the fine object. With `segment_levels` that is the only parent. With two unrelated segmentations it is the
best overlap, which is a reasonable choice but not a true containment.

### How does an object differ from its parent? `super_object_features`

```python
sup = obia.super_object_features(fine.labels, coarse.labels, scene.raster,
                                 extra_layers=obia.spectral_layers(scene.raster))
sup[["parent_id", "parent_area_px", "rel_area_of_parent", "diff_to_parent_ndvi"]].head(5).round(3)
```

```text
        parent_id  parent_area_px  rel_area_of_parent  diff_to_parent_ndvi
obj_id
1               1           891.0               0.398                0.003
2               2          1610.0               0.615                0.004
3               3          4447.0               0.024                0.015
4               3          4447.0               0.047               -0.005
5               3          4447.0               0.023                0.004
```

| Column | Meaning | Reading it |
|---|---|---|
| `parent_id` | Coarse object id | Join key to coarse-level tables |
| `parent_area_px` | Area of the parent in pixels | Large parent: the object is part of something big |
| `rel_area_of_parent` | Own area ÷ parent area | Near 1: the object *is* its parent. Near 0: one small piece of a large whole |
| `diff_to_parent_<layer>` | Own mean − parent mean, for every band and extra layer | Near 0: typical for its parent. Large: stands out inside it |

Objects 3, 4 and 5 above share parent 3 (4,447 pixels) and each is 2–5 % of it: they are fragments of one big thing, and
their `diff_to_parent_*` values are small. A fine object that is a *small share of its parent and very different from it*
(for example a bright roof inside a dark parent) is the interesting case.

Use these as columns in rules or in `ObjectClassifier`; join them on the object id:

```python
feats = feats.join(sup)            # both are indexed by obj_id
rules = obia.RuleBasedClassifier([("lone_object", "rel_area_of_parent < 0.1 and abs(diff_to_parent_ndvi) > 0.1")])
```

### What is a super-object made of? `sub_object_fraction`

The other direction: classify the **fine** level, then say how much of each coarse object each class covers.

```python
frac = obia.sub_object_fraction(fine_result.classes, fine_feats["area_px"], pm, "cropland")
frac.head(8).round(3)
```

```text
parent_id
1    0.000
2    1.000
3    0.000
4    0.984
5    0.000
6    0.000
7    0.000
8    0.100
Name: area, dtype: float64
```

Read: coarse object 2 is 100 % cropland (by area), 4 is 98 %, 8 is 10 %. Typical uses: call a coarse object "cropland" if
it is more than 70 % cropland objects; flag "mixed" parents (every class below 70 %) for review; compute the built-up
share of a block. `sub_object_fraction(fine_classes, fine_area, parents, class_name)` takes the three Series aligned on
the fine object id; objects without a parent are ignored.

## Context

### Neighbour classes: `class_context`

```python
first = rules.classify(feats)                              # any first classification
ctx = obia.class_context(seg.labels, first.classes)
ctx.filter(like="rel_border").round(2).head(3)
```

```text
        rel_border_bare_field  rel_border_building  rel_border_cropland  rel_border_forest  rel_border_road  rel_border_unclassified  rel_border_water
obj_id
1                        0.03                0.00                 0.88               0.09             0.00                     0.00               0.0
2                        1.00                0.00                 0.00               0.00             0.00                     0.00               0.0
3                        0.02                0.07                 0.37               0.25             0.14                     0.15               0.0
```

For every class `c` it gives two columns:

| Column | Meaning |
|---|---|
| `rel_border_<c>` | Share of the object's border with other objects (by shared border length) that touches objects of class `c`. Over all classes it sums to exactly 1 for any object that has a neighbour; the stretch of border on the image frame is not counted here (see `rel_border_image_edge` in the [feature table](features.md#context-neighbours)) |
| `n_nb_<c>` | Number of neighbours of class `c` |

Object 1 above has 88 % of its border against cropland. Object 2 is completely surrounded by bare field. Class names are
cleaned like feature names (`bare_field` stays `bare_field`).

Two things to keep in mind:

- The context is **only as good as the first classification**. If the first pass is wrong about the neighbours, context
  features pass that error on.
- Feed context back in a **second pass**. A rule that uses `rel_border_road` needs `class_context` to have been computed
  from a classification that already contains roads.

## A worked example: slivers on field edges

On the demo scene, scale 20, a first set of rules (water, forest, cropland, road by `elongation > 8`, building, bare field, in that order)
scores **0.756** overall accuracy against the true objects. The confusion matrix shows the main error:

```text
              bare_field  building  cropland  forest  road  unclassified  water
cropland               0         0         8       8     0             1      0
```

Eight cropland objects were called **forest**. Printing them shows why:

```text
 obj  truth     area_px  std_nir  mean_ndvi
   5  cropland     25     17.7      0.44
  10  cropland     79     16.6      0.67
  17  cropland     27     29.0      0.49
  32  cropland    117     19.9      0.59
   9  forest     1734     13.7      0.76      <- real forest
  37  forest     1384     14.1      0.62      <- real forest
```

These are **slivers** (20–117 pixels) that sit across a boundary between two fields. They contain pixels from both, so
their `std_nir` is high, which the forest rule reads as "textured". Real forest is both textured *and* large (1,300+ pixels).

**Fix 1: add a size condition** (this uses the feature table only):

```python
("forest", "mean_ndvi > 0.4 and std_nir > 8 and area_px > 300")
```

Accuracy rises from **0.756 to 0.844** (kappa 0.793), area-weighted from 0.968 to 0.977.

**Fix 2: give small leftovers the class they mostly touch** (this is where context comes in):

```python
first = obia.RuleBasedClassifier(rules_with_size_condition).classify(feats)    # the result of fix 1
ctx = obia.class_context(seg.labels, first.classes)
feats2 = feats.join(ctx)
small = (feats2.area_px < 130) & ~first.classes.isin(["building", "road"])
neighbour_cols = ["rel_border_bare_field", "rel_border_cropland"]
relabel = feats2.loc[small, neighbour_cols].idxmax(axis=1).str.replace("rel_border_", "")
classes = first.classes.copy()
classes.loc[relabel.index] = relabel
```

Accuracy rises to **0.867** (kappa 0.818). Of the 15 small objects considered, 5 actually changed class, and the net
gain is **one more correct object out of 45** (39 instead of 38). Look at what happened, not only the headline: the
confusion matrix shows cropland slivers that were `unclassified` are now assigned, but two real cropland objects became
`bare_field` because they touched more bare field than cropland. Context changed *which* objects are wrong, and on balance
slightly fewer. That is the typical outcome: context trims errors at the edges of classes; it does not repair a rule that
is wrong in the middle.

```{important}
These thresholds were chosen **by looking at this same scene**, so the 0.756 → 0.844 → 0.867 sequence shows what the fix
does, not what you can expect on new imagery. To get an honest number, write the rules on one scene and assess them on
another ([Accuracy](accuracy.md#the-training-data-trap)).
```

## Which is the right tool?

| Mistake you see | Try | Why |
|---|---|---|
| Tiny objects with strange values on class borders | A size condition; or raise `scale` so slivers merge | Slivers are an artefact of segmentation, so fix them there first |
| A class that is defined by what surrounds it (island, courtyard, field next to a river) | `class_context` + second pass | Neighbour class is the defining property |
| A coarse region should be named by what it contains | `sub_object_fraction` | Majority of sub-objects decides |
| A fine object should be judged relative to the larger whole | `super_object_features` | `diff_to_parent_*` shows deviation from the parent |
| Objects of one kind need a finer scale than another | Two levels with `segment_levels`, classify each | One scale rarely fits both buildings and fields |
| Same error everywhere | Fix the rule or the training samples | Context cannot repair a systematic mistake |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `class_context` has no column for a class | That class does not occur in the classification | Pass `class_names=[...]` so the columns exist with zeros, or skip the rule |
| `rel_border_*` columns do not sum to 1 | You passed `class_names` that leave out some classes, or the object has no neighbours | Include every class, or accept that the rest goes to the omitted ones |
| `parent_map` gives a parent that covers only part of the object | The two segmentations are not nested | Use `segment_levels` to get a true hierarchy |
| `super_object_features` has no row for some object | Background `0` is never an object | Make sure both rasters use ids ≥ 1 for objects |
| Re-labelling changes many correct objects | Threshold too generous | Look at the confusion matrix before and after; tighten the condition |
| Context rule seems to do nothing | It was computed from a first pass that did not contain the neighbour class | Add the neighbour class to the first pass, then recompute |
| `NaN` in a joined table | `feats.join(sup)` on a different level's ids | Join fine-level features with fine-level `sup`; coarse tables use `parent_id` |
| Result differs after changing the first pass | Context depends on it | Expected: context inherits the first pass's errors |

```{seealso}
[Segmentation](segmentation.md) · [Object features](features.md) · [Classification](classification.md) ·
[Accuracy assessment](accuracy.md)
```
