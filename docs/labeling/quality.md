# Quality checks and cleaning

**Class:** `pygeovision.labeling.LabelQualityAssessor` · **CLI:** `pygeovision label quality` · **Client:** `client.labeling.quality(path)`

## Why use it

Before you spend hours training on a label raster, a minute of checking can show that it is empty, speckled with
single-pixel dust, mostly no-data, or has one class that makes up 99.9% of the pixels. `LabelQualityAssessor` looks at the
raster and tells you, with a score, a letter grade and plain-language advice. It can also remove specks and fill small holes.

```{important}
This is a **sanity check, not a measure of accuracy.** It never compares the labels to the ground or to the imagery
(`image_path` is accepted but not used). A perfectly clean raster can be completely wrong; a messy one can be right. The
score tells you about the *shape of the data*, not whether buildings are where the label says.
```

## What the checks measure

| Check | Question | Score | Flag |
|---|---|---|---|
| `class_balance` | How lopsided are the classes? | `1 / (1 + ratio/10)`, where ratio = largest class ÷ smallest class | `ok` below 20×, `warning` below 100×, `critical` above |
| `coverage` | How much of the raster has a label (not 255/nodata)? | `min(1, valid_fraction × 2)` | `warning` if valid share ≤ 50% |
| `slivers` | How many specks are there? A speck is a connected patch smaller than `max(4 px, 1 m²)` | falls as specks per 0.1% of pixels grow | `warning` at 100 or more specks |
| `connectivity` | What share of the foreground sits in real objects (16 px or more)? | that share | `warning` below 80% |
| `boundaries` (off by default, needs OpenCV) | How compact are the shapes? (1 = circle, 0.1 = ragged) | `min(1, 2 × quotient)` | `warning` if quotient ≤ 0.05 |

The overall `quality_score` is the **mean** of the scores of the checks that ran. Grade: A ≥ 0.90, B ≥ 0.75, C ≥ 0.60,
D ≥ 0.40, F below. A raster with **no foreground at all** scores 0 and gets a recommendation, however clean it looks.

Class imbalance often reflects the world (buildings are 5% of a city). It lowers the score gently and is advice for
[the loss function](../training/losses.md), not a defect.

## How to use

```python
from pygeovision.labeling import LabelQualityAssessor

qa = LabelQualityAssessor()                       # classes inferred from the raster
report = qa.assess("labels/osm.tif")
print(report["quality_grade"], report["quality_score"])
for line in report["recommendations"]:
    print(" -", line)

if report["quality_score"] < 0.8:
    qa.clean("labels/osm.tif", "labels/osm_clean.tif", min_area_m2=20)
```

```console
$ pygeovision label quality labels/osm.tif --html
$ pygeovision label quality labels/osm.tif --check slivers --check connectivity --num-classes 11
```

### `LabelQualityAssessor(num_classes=None, ignore_index=255)`

| Argument | Meaning |
|---|---|
| `num_classes` | `None` (default): the classes are whatever values occur. A number `N` means classes `0…N-1`; any other value (except `ignore_index`) is reported in `unexpected_values` instead of being silently dropped |
| `ignore_index` | The "no label" value, default 255. The raster's own nodata value is also treated as no label |

### `assess(label_path, image_path=None, checks=None)`

`checks` is any subset of `class_balance`, `slivers`, `coverage`, `connectivity`, `boundaries`. The default runs all but
`boundaries`. An unknown name raises `ValueError`. `slivers` and `connectivity` need `scipy`; `boundaries` needs
`opencv-python`. If they are missing the check is reported as `skipped` with the reason and does not affect the score.

Speck size is measured in **real ground area**. A raster in degrees is converted to square metres at its centre latitude,
so "1 m²" means one square metre whatever the CRS.

## What you get

Real output on the small OSM test raster from the [OSM page](osm.md) (112 × 72 pixels, ≈ 99 m² per pixel):

```python
{'quality_score': 0.791, 'quality_grade': 'B',
 'classes': [0, 1, 2, 4, 7], 'unexpected_values': [], 'pixel_area_m2': 98.7684, 'foreground_pixels': 5373,
 'checks': {
   'class_balance': {'class_pct': {0: 33.37, 1: 1.25, 2: 3.72, 4: 1.19, 7: 60.47},
                     'imbalance_ratio': 50.3, 'score': 0.166, 'status': 'warning'},
   'coverage':      {'valid_fraction': 1.0, 'score': 1.0, 'status': 'ok'},
   'slivers':       {'n_slivers': 0, 'sliver_threshold_px': 4, 'score': 1.0, 'status': 'ok'},
   'connectivity':  {'n_components': 5, 'share_in_objects': 1.0, 'min_object_px': 16, 'score': 1.0, 'status': 'ok'}},
 'recommendations': ['Class imbalance ratio 50x -- use a weighted/overlap loss (see docs/training/losses.md) or oversample the minority class']}
```

Read it as: the shapes are solid objects, with no dust and full coverage. The only issue is that the largest class (60% residential)
dwarfs the smallest (1.2% vegetation). Nothing is wrong with the raster, and the advice is about training.

An unreadable file gives `{"error": "...", "quality_score": 0.0}` without raising.

`report_html(report)` returns a small standalone HTML page (grade, check table, recommendations); the CLI's `--html` writes it
beside the label file as `<stem>_quality.html`.

## `clean`: remove specks, fill holes

```python
result = qa.clean("labels/osm.tif", "labels/osm_clean.tif",
                  remove_slivers=True, min_area_m2=5.0, fill_holes=True, smooth=True)
```

| Argument | Default | Meaning |
|---|---|---|
| `remove_slivers` | `True` | Delete connected patches smaller than `min_area_m2` (set to background 0) |
| `min_area_m2` | 5.0 | Smallest patch kept, in real square metres |
| `fill_holes` | `True` | Fill background holes completely inside an object |
| `smooth` | `True` | A 3 × 3 closing: merges gaps of one or two pixels (needs OpenCV) |

Rules the cleaner follows:

- Class 0 (background) and 255 (no data) are never edited, except that holes and gaps are filled **from background**: a
  foreground class may grow into background, never into another class and never into no-data.
- Removed specks become **background (0)**, not no-data.

Result: `success`, `output_path`, `n_components_removed`, `min_component_px`, `before_label_pixels`,
`after_label_pixels`, `holes_filled_px`, `smoothed_px`. The pixel counts exclude no-data, so you can see exactly how much
was changed. If `after` is far below `before`, your `min_area_m2` is too big for your resolution.

```{warning}
`fill_holes` also fills **genuine courtyards** (a ring-shaped building becomes solid) and `smooth` merges objects that
are closer than about two pixels (two houses a metre apart become one). Turn those off for labels where that matters:
`qa.clean(..., fill_holes=False, smooth=False)`.
```

## What to do when…

| Recommendation / symptom | What it means | Do this |
|---|---|---|
| "Class imbalance ratio 50x -- use a weighted/overlap loss…" | One class dwarfs another | [Loss functions and imbalance](../training/losses.md); sample chips that contain the rare class |
| "N specks smaller than K px -- run clean()" | Dust, usually from vector edges or SAM | `clean()` |
| "Only 12% of the foreground sits in objects of ≥ 16 px" | The label is mostly dust: likely the wrong source, or the wrong resolution | Check the source and the grid before cleaning |
| "Less than 50% of pixels carry a label" | Large no-data area | Expand coverage or choose chips with more labelled area; the 255 pixels are ignored in training, so this is fine only if enough remains |
| "The raster has no foreground pixels" | Nothing was labelled | The source returned nothing for this area. Check the box and the source |
| "Values [12, 13] are outside 0..9 and were not counted" | Wrong `num_classes`, or an un-remapped source | Raise `num_classes`, leave it `None`, or remap |
| Score is low but the labels look fine | Heavy imbalance or a legitimately sparse class | Expected for rare classes; judge by eye |
| Score is high but the labels are shifted from the imagery | Quality checks never look at imagery | Overlay on imagery |
| `skipped` + `pip install scipy` | Missing package | Install it |

## Typical use in a workflow

1. Make labels ([OSM](osm.md), [footprints](buildings.md), [fusion](pipeline.md)).
2. `assess` each; read the recommendations.
3. `clean` if specks are the issue.
4. **Look at the result in QGIS over the imagery** (the one check no script replaces).
5. Re-assess the cleaned file; the score should not fall.

```{seealso}
[Preparing training data](../training/data.md) · [Combining sources](pipeline.md) · [Active learning](active.md)
```
