# Combining label sources

**Class:** `pygeovision.labeling.AutoLabelPipeline` · **Client:** `client.labeling.pipeline(bbox, sources=[...], output_dir=...)`

## Why use it

Every source is wrong somewhere, and **usually in different places**. OSM misses buildings in new suburbs; Microsoft
merges terraces; Google misses small roofs. When two or three independent sources say "building" for the same pixel, it is
much more likely to be one. `AutoLabelPipeline` runs several sources over one area, puts them on one grid and votes.

It also saves you from writing the same glue each time: one call runs every source, keeps going if one fails, writes the
per-source rasters, fuses what can be fused, and scores the result.

## The one rule: only fuse labels that mean the same thing

A vote between two rasters is only meaningful if their numbers mean the same. Here is what that gives:

| Sources | Fused? | Why |
|---|:-:|---|
| OSM `["buildings"]` + Microsoft + Google | ✓ | All are 0 / 1 = "building" |
| OSM with several categories + ESA WorldCover | ✗ | OSM: 1 = building, 2 = road. ESA: 10 = trees, 50 = built-up. A vote would mix unrelated classes |
| Anything + SAM or foundation | ✗ | Their numbers are arbitrary ids (object 7, cluster 3) |

When fusion is not meaningful the pipeline **does not do it**. It writes the per-source rasters, leaves `fused_path` as
`None` and says why in `fusion_skipped_reason`. In older versions the pipeline voted anyway and returned a confident
looking, meaningless raster.

To fuse land-cover sources, first remap each to the same scheme (for example "built-up = 1, everything else = 0") with the
labelers' `remap_classes`, then run them as separate labelers and combine the 0/1 rasters yourself or through this class.

## How to use

```python
from pygeovision.labeling import AutoLabelPipeline

pipe = AutoLabelPipeline(
    sources=["osm", "microsoft_buildings", "google_buildings"],
    fusion="majority_vote",
    reference_raster="data/lagos_s2.tif",     # all sources land on this grid
)
result = pipe.run(
    bbox=(3.35, 6.45, 3.45, 6.55),
    output_dir="labels/lagos/",
    categories=["buildings"],                 # for OSM; gives 0/1
)
print(result["sources_succeeded"], result["sources_failed"])
print(result["fused_path"], result["fusion_skipped_reason"])
print(result["quality"]["quality_grade"])
```

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `sources` | `["osm", "esa_worldcover"]` | Any of `osm`, `microsoft_buildings`, `google_buildings`, `esa_worldcover`, `dynamic_world`, `sam_auto`, `foundation`. An unknown name raises `ValueError` listing the valid ones |
| `fusion` | `"majority_vote"` | `majority_vote`, `union`, `intersection`, `priority`. Anything else raises `ValueError` |
| `quality_threshold` | 0.6 | Stored on the object; informational |
| `reference_raster` | `None` | Put every source on this grid. **Required** for `sam_auto` and `foundation` (they segment an image, not a box) |

`run(bbox, output_dir="./labels/", categories=None, assess_quality=True)`. `categories` goes to OSM (default buildings,
roads, water).

The default `sources=["osm", "esa_worldcover"]` cannot be fused (different alphabets), so a plain
`AutoLabelPipeline().run(bbox)` gives you two rasters and an explanation. That is a safe default, not a bug.

### Fusion modes

Binary modes (values 0/1):

| Mode | A pixel is "building" when… | Choose when |
|---|---|---|
| `majority_vote` | most sources say so; ties go to the source listed **first** | Default; balanced |
| `union` | **any** source says so | You would rather miss nothing (then clean up) |
| `intersection` | **all** sources say so | You want only certain examples |
| `priority` | the first source (in `sources` order) with a non-zero value has one | One source is trusted, others fill gaps |

A pixel that **no source covered** is `255` (no data) in the output, not 0.

## What you get

```text
labels/lagos/osm.tif                 one raster per source that worked
labels/lagos/ms_buildings.tif
labels/lagos/google_buildings.tif
labels/lagos/fused_labels.tif        uint8, nodata=255, tags: fusion mode, source files
```

Result keys (real output, three offline 0/1 sources on one UTM grid, each a 20 × 20 pixel square, offset from one another by 2–4 pixels):

```python
{'success': True,
 'sources_succeeded': ['osm', 'microsoft_buildings', 'google_buildings'],
 'sources_failed': [],
 'fused_path': 'labels/fuse/fused_labels.tif',
 'fusion_skipped_reason': None}
# fused raster: {0: 3208, 1: 392}      quality: 0.888, grade B
```

| Key | Meaning |
|---|---|
| `success` | At least one source wrote a raster |
| `sources_succeeded` / `sources_failed` | By source. A source that "succeeded" but wrote no file (Earth Engine export) is counted as failed |
| `label_paths` | The rasters written |
| `fused_path` | The fused raster, or the single raster if only one source worked, or `None` |
| `fusion_skipped_reason` | In words, when sources worked but were not fused |
| `partial_results` | Each source's own result dict with its `error` |
| `quality`, `quality_by_source` | [Quality reports](quality.md) for the fused raster and for each source |

One failing source does not stop the others. If all but one fail, that one's raster is returned as `fused_path` and the
others are listed in `sources_failed`. Look at `partial_results[source]["error"]` for each.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `fusion_skipped_reason: "these sources use different label alphabets (osm has values [0, 1, 2]; esa_worldcover has values [10, 50])…"` | Mixed schemes | Use `categories=["buildings"]` for OSM, or remap |
| `fusion_skipped_reason: "['sam_auto'] produce arbitrary ids…"` | SAM or foundation in the list | Run them separately and use [`assign_classes`](naming-segments.md) |
| `sam_auto requires reference_raster` | They need an image | Pass `reference_raster` |
| `ValueError: Unknown source(s)` | Typo | Use the names in the table |
| One source `success=False` | Network, coverage, missing package | Read `partial_results[name]["error"]` |
| Fused result looks like one source only | Only one succeeded | Check `sources_succeeded` |
| Fused raster has a stripe of 255 | A source covers only part of the box (Google cell limit, partial download) | Check `n_*_failed` in that source's result |
| Majority vote with two sources | A tie everywhere they disagree | Use three sources, or `union`/`intersection` deliberately |

```{seealso}
[Quality and cleaning](quality.md) · [OSM](osm.md) · [Building footprints](buildings.md) ·
[Land cover](landcover.md)
```
