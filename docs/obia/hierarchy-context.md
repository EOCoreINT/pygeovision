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

levels = obia.segment_levels(rd, scales=[10, 30])   # one merging run, paused twice
fine, coarse = levels[10.0], levels[30.0]
print(fine.n_objects, coarse.n_objects)                        # many fine objects, fewer coarse ones
```

Because both levels come from the same run, **every coarse object is an exact union of fine objects** (every fine object lies inside
exactly one coarse object). Two independent `segment` calls at
different scales do not guarantee that. See [Segmentation](segmentation.md#several-scales-at-once-segment_levels).

### Which coarse object does a fine object belong to? `parent_map`

```python
pm = obia.parent_map(fine.labels, coarse.labels)
pm.head(5)
```

The result is a Series indexed by fine object id, whose value is the id of the coarse object (`coarse`).

`parent_map(fine_labels, coarse_labels)` returns a Series indexed by fine id whose value is the coarse object that covers
**most** of the fine object. With `segment_levels` that is the only parent. With two unrelated segmentations it is the
best overlap, which is a reasonable choice but not a true containment.

### How does an object differ from its parent? `super_object_features`

```python
sup = obia.super_object_features(fine.labels, coarse.labels, rd,
                                 extra_layers=obia.spectral_layers(rd))
sup[["parent_id", "parent_area_px", "rel_area_of_parent", "diff_to_parent_ndvi"]].head(5).round(3)
```

One row per fine object, with the columns below.

| Column | Meaning | Reading it |
|---|---|---|
| `parent_id` | Coarse object id | Join key to coarse-level tables |
| `parent_area_px` | Area of the parent in pixels | Large parent: the object is part of something big |
| `rel_area_of_parent` | Own area ÷ parent area | Near 1: the object *is* its parent. Near 0: one small piece of a large whole |
| `diff_to_parent_<layer>` | Own mean − parent mean, for every band and extra layer | Near 0: typical for its parent. Large: stands out inside it |

Fine objects that share a large parent and are each a few percent of it are fragments of one big thing, and their
`diff_to_parent_*` values are small. A fine object that is a *small share of its parent and very different from it* (for
example a bright roof inside a dark parent) is the interesting case.

Use these as columns in rules or in `ObjectClassifier`; join them on the object id:

```python
feats = feats.join(sup)            # both are indexed by obj_id
rules = obia.RuleBasedClassifier([("lone_object", "rel_area_of_parent < 0.1 and abs(diff_to_parent_ndvi) > 0.1")])
```

### What is a super-object made of? `sub_object_fraction`

The other direction: classify the **fine** level, then say how much of each coarse object each class covers.

```python
frac = obia.sub_object_fraction(fine_result.classes, fine_feats["area_px"], pm, "cropland")
```

The result is a Series indexed by coarse object id: the share of each coarse object's area that is covered by fine objects
of that class (1.0 = all of it, 0.1 = a tenth). Typical uses: call a coarse object "cropland" if
it is more than 70 % cropland objects; flag "mixed" parents (every class below 70 %) for review; compute the built-up
share of a block. `sub_object_fraction(fine_classes, fine_area, parents, class_name)` takes the three Series aligned on
the fine object id; objects without a parent are ignored.

## Context

### Neighbour classes: `class_context`

```python
first = rules.classify(feats)                              # any first classification
ctx = obia.class_context(seg.labels, first.classes)
ctx.filter(like="rel_border").round(2).head(3)      # one rel_border_<class> column per class
```

For every class `c` it gives two columns:

| Column | Meaning |
|---|---|
| `rel_border_<c>` | Share of the object's border with other objects (by shared border length) that touches objects of class `c`. Over all classes it sums to exactly 1 for any object that has a neighbour; the stretch of border on the image frame is not counted here (see `rel_border_image_edge` in the [feature table](features.md#context-neighbours)) |
| `n_nb_<c>` | Number of neighbours of class `c` |

A row such as `rel_border_cropland = 0.88` means that 88 % of the object's border is against cropland objects; an object
completely surrounded by one class has 1.0 for it. Class names are cleaned like feature names (`bare_field` stays `bare_field`).

Two things to keep in mind:

- The context is **only as good as the first classification**. If the first pass is wrong about the neighbours, context
  features pass that error on.
- Feed context back in a **second pass**. A rule that uses `rel_border_road` needs `class_context` to have been computed
  from a classification that already contains roads.

## A worked example: slivers on field edges

A common error in field scenes: **slivers**, small objects that sit across the boundary between two fields. They contain
pixels from both, so their `std_nir` is high, and a rule that reads high `std_nir` as "textured, therefore forest" calls them
forest. Real forest is both textured *and* large. To find out whether this happens in your scene, print the objects that a rule
claimed wrongly (against your reference) with their `area_px`, `std_nir` and `mean_ndvi`: slivers show up as small areas
with the "wrong" texture.

**Fix 1: add a size condition** (this uses the feature table only):

```python
("forest", "mean_ndvi > 0.4 and std_nir > 8 and area_px > 300")
```

Choose the area threshold from your own table, not from this page: it is a few times the size of a sliver and well below the
size of a stand of trees at your resolution.

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

Then look at what changed, not only at the score: compare the confusion matrices before and after. Context trims errors at the
edges of classes (slivers take the class they sit in), but it can also flip a real small object that happens to touch
more of another class. It does not repair a rule that is wrong in the middle of a class.

```{important}
Thresholds chosen **by looking at a scene** fit that scene. The improvement you see on it shows what the fix does, not what you
can expect on new imagery. For an honest number, write the rules on one area and assess them on another, against reference
points they were not tuned on ([Accuracy](accuracy.md#the-training-data-trap)).
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
