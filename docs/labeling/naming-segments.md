# Naming segments with `assign_classes`

**Function:** `pygeovision.labeling.assign_classes`

## Why use it

[SAM](sam.md) gives you good **outlines** with no names. [Foundation clusters](foundation.md) give you **groups** with no
names. [OSM](osm.md), [WorldCover](landcover.md) and hand-labelled areas give you **names** but crude outlines. Putting
them together gets the best of both: sharp object shapes carrying classes from a map.

`assign_classes` lays the class map under the segment map. Each segment takes the class that **most of its pixels
already have** in the class map. If no class holds enough of the segment, the segment is marked `255` (ignore) rather
than forced into a class it may not belong to.

```text
SAM segment 7:  400 px  →  under it in OSM: 360 px "building" (1), 40 px "background" (0)
                          360 / 400 = 90% ≥ min_share (0.5)  →  whole segment becomes class 1

SAM segment 9:  100 px  →  under it in OSM: 55 px "building", 45 px "road"
                          55% ≥ 0.5 → class 1.      With min_share=0.6 → 255 (ignore)
```

## How to use

```python
from pygeovision.labeling import SAMAutoLabeler, OSMLabeler, assign_classes

SAMAutoLabeler(model="sam-vit-base").auto_label("data/aerial.tif", "labels/sam.tif", rgb_bands=(1, 2, 3))
OSMLabeler().label(bbox, ["buildings", "roads"], "labels/osm.tif", reference_raster="data/aerial.tif")

r = assign_classes("labels/sam.tif", "labels/osm.tif", "labels/named.tif", min_share=0.6)
print(r["n_segments"], r["n_unlabelled"], r["mapping"], r["class_pixels"])
```

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `segments_path` | required | Raster of segment ids (SAM `auto_label` output, a cluster map, …) |
| `reference_path` | required | Class raster to learn from. If its grid differs, it is resampled onto the segments' grid with nearest neighbour. Its own nodata is not counted as a vote |
| `output_path` | required | `uint8` class raster on the segments' grid |
| `min_share` | 0.5 | Fraction of a segment's **voting** pixels the winning class must reach, in `(0, 1]` |
| `ignore_segment_ids` | `(0, 255)` | Ids meaning "no segment" (SAM uses 0; the cluster labeler uses 255 for no data) |
| `outside_value` | 0 | Class written where there is no segment |
| `unlabelled_value` | 255 | Value for segments where no class reaches `min_share` |
| `reference_nodata` | `None` | Override the reference raster's nodata value |

## What you get

Real output from a small synthetic test (three segments; segment 1 sits fully on class 1, segment 2 is half on class 2 and
half on background, segment 3 sits on background):

```python
{'success': True, 'output_path': 'labels/named.tif', 'n_segments': 3, 'n_unlabelled': 1,
 'mapping': {1: 1, 2: 255, 3: 0}, 'class_pixels': {0: 3100, 1: 400, 255: 100}}
```

| Key | Meaning |
|---|---|
| `mapping` | `segment id → class` (or 255). This is how you audit decisions |
| `n_segments` | Segments looked at |
| `n_unlabelled` | Segments set to 255 |
| `class_pixels` | Pixels per output class |

The output is a normal label raster (`uint8`, nodata 255) that the [trainer](../training/data.md) can use directly.

## Choosing `min_share`

| Value | Effect | Use when |
|---|---|---|
| 0.5 | Majority wins | The class map is good and you want few holes |
| 0.6–0.8 | Only confident segments are named | The class map is noisy or incomplete (OSM in a thinly mapped area) |
| 1.0 | Only segments entirely inside one class | You want the cleanest examples and can afford to lose many |

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| Almost everything is `255` | `min_share` too high, or the class map and segments are misaligned | Check both rasters on the same grid in QGIS; lower `min_share` |
| Everything becomes class 0 | The class map has few features in this area, so background dominates every segment | A segment mostly on "background" is named background. Use a class map that covers your area; or ignore class 0 by passing `reference_nodata=0` |
| A building segment is named road | The road line overlaps it | Raise `min_share`; draw order in OSM already puts buildings over roads |
| `ValueError: min_share must be in (0, 1]` | Out-of-range value | Use 0.5–0.9 |
| The reference differs in CRS or resolution | Different grid | It is resampled by nearest neighbour; for exact results make both with `reference_raster` set to the same imagery |
| Segments from several scenes | Different id spaces | Run each scene separately, then merge the *named* outputs |

```{admonition} The naming is only as good as the class map
:class: warning
If OSM has not mapped a house, SAM will outline it and `assign_classes` will call it background. It does not add
knowledge; it **transfers** the class map's knowledge onto better shapes, including its gaps and mistakes. Use a
[quality check](quality.md) and spot-check the result over the imagery.
```
