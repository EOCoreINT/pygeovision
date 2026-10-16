# Vegetation change in central London: a crop-health pipeline pointed at a city

`crop_health` compares NDVI on two dates and flags the pixels where it fell by more than a threshold. It was written for fields.
This run points it at a 19.3 km² box of central London, on purpose, because that is the more instructive way to read the report:
**every number in it is correct, and almost none of them says what the pipeline's name suggests.** The page goes through what the
5.8 % actually is, what it is a share of, why the two dates are a problem, and how to turn the result into something
you can use, with the land-cover run from the [first London page](london-landcover-habitat.md).

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`crop_health_report.pdf`, PyGeoVision 2.2.0, generated
  2026-10-15 18:53 UTC). The shares, the pixel areas and the mean change are re-derived in `tests/test_case_studies.py`.
- **Not re-run:** the two Sentinel-2 scenes were not downloaded again.
- **No ground truth.** Nothing in the report says why any pixel changed. Where this page offers a reason, it says "a hypothesis".
- **Not an assessment of crops:** there are essentially no crops in the box (WorldCover has one cropland pixel, see the
  [land-cover page](london-landcover-habitat.md)).
```

## The result at a glance

| Figure | Value |
|---|---:|
| Valid area flagged ("NDVI dropped") | **5.8 %** of the *valid* area |
| The same flagged area as a share of the whole box | 2.6 % (49.6 ha of 1,940.1 ha) |
| Valid share of the box (cloud-free on both dates) | **44.2 %** |
| Threshold | NDVI fall greater than 0.20 |
| Mean NDVI, before / after | 0.287 / 0.249 |
| Mean change | -0.038 (median -0.029) |
| Polygons in the vector layer | 436 |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -0.1500, 51.4700, -0.1000, 51.5200 (central London, by its coordinates) |
| Output grid | 421 × 421 = 177,241 pixels, EPSG:4326 (the raster extent is 19.40 km² against 19.32 km² for the box) |
| Before | `S2B_MSIL2A_20240520T110619_R137_T30UXC_20240520T151432`, 20 May 2024 |
| After | `S2A_MSIL2A_20240823T110621_R137_T30UXC_20240823T172700`, 23 August 2024 |
| Bands | red and NIR, from Sentinel-2 L2A surface reflectance |
| Cloud mask | on; a pixel masked on either date is empty |
| Threshold | `anomaly_threshold = 0.2` (the default is 0.15) |

## Run it

```bash
pygeovision channel crop_health \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date-before 2024-05 \
    --date-after 2024-08 \
    --param anomaly_threshold=0.2 \
    --output out/crop
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "crop_health",
    bbox=(-0.15, 51.47, -0.10, 51.52),
    date_before="2024-05",
    date_after="2024-08",
    anomaly_threshold=0.2,
    output_dir="out/crop",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["pct_flagged"], result.stats["valid_fraction"])
```

Each date is a month (`YYYY-MM`). The pipeline chose one scene for each, and the report lists them.

## What the pipeline did

```{figure} ../_static/img/tree-case-london.svg
:alt: One central-London box and its five pipelines: land cover, biodiversity hotspot, building footprints, road network and crop health, with the box and its coordinates at the root
:width: 100%
:class: pgv-figure

The whole London set from one box. This page is the last branch.
```

1. Find a scene in each month and download red and NIR with the cloud and shadow pixels left empty.
2. NDVI = (NIR - red) / (NIR + red), on both dates.
3. Change = after - before. A pixel is flagged when before - after is **greater than** the threshold.
4. Every percentage and area is over the pixels that are valid on **both** dates.

## The result

```{figure} ../_static/img/case-studies/ndvi-drop-map.webp
:alt: Map of flagged pixels in red over a pale grey background; the red patches are small and scattered, with the larger clusters in the lower (southern) half and along the left edge, and with large pale areas that have no data
:width: 100%
:class: pgv-figure

Flagged pixels (red). The slightly darker grey is valid and not flagged; the lighter background is no data.
```

| Class | Value | Pixels | Area (ha) | Share of valid area | Share of the box |
|---|---:|---:|---:|---:|---:|
| Not flagged | 0 | 73,773 | 807.5 | 94.2 % | 41.6 % |
| NDVI dropped | 1 | 4,528 | 49.6 | 5.8 % | 2.6 % |

```python
# check: the flagged share is a share of the valid area, and the valid area is under half of the box
flagged, valid, total = 4_528, 78_301, 177_241
print(f"{100 * flagged / valid:.1f} % of the valid pixels, {100 * flagged / total:.1f} % of the box; valid share {100 * valid / total:.1f} %")
print(f"{flagged * 1940.1 / total:.1f} ha flagged, {flagged / 436:.1f} pixels per polygon on average")
```

```text
5.8 % of the valid pixels, 2.6 % of the box; valid share 44.2 %
49.6 ha flagged, 10.4 pixels per polygon on average
```

The valid area is 78,301 pixels (857.1 ha), of which 73,773 are not flagged and 4,528 are.

The polygon count is worth a pause: 436 polygons for 4,528 pixels means a typical patch is about ten 10 m pixels, around a tenth of a
hectare. At 10 m, a single tree crown, a lawn or a roof edge is that size, so many of the flagged patches are one object.

### The change map

```{figure} ../_static/img/case-studies/ndvi-drop-change.webp
:alt: Map of NDVI change from after minus before. Most valid pixels are pale yellow (a small fall); a band of pale green (increase) runs from the lower left through the middle to the top, and small dark-red patches (large falls) are scattered, with a cluster in the south
:width: 100%
:class: pgv-figure

NDVI after minus before. Yellow is a small fall, red a large fall, green a rise.
```

The most striking thing in the change map is not a loss but a **band of increase** that runs through the box. Its shape matches the
river on the [land-cover map](london-landcover-habitat.md). Water is not vegetation, and NDVI over water is a small or negative number
that can move with turbidity, algae and the state of the tide. That is a hypothesis, not something the report shows, but it is
the kind of thing a vegetation index does when nothing masks the water.

```{figure} ../_static/img/case-studies/ndvi-drop-change-hist.webp
:alt: Histogram of NDVI change centred just below zero, a narrow hump between about -0.15 and 0.1, and long thin tails on both sides; the 5th to 95th percentile range is shaded
:width: 100%
:class: pgv-figure

The distribution of change. The flag is the left tail, beyond -0.2.
```

| Statistic of NDVI change | Value |
|---|---:|
| Minimum / maximum | -0.872 / +0.759 |
| 5th / 25th percentile | -0.216 / -0.066 |
| Median / mean | -0.029 / -0.038 |
| 75th / 95th percentile | +0.004 / +0.097 |
| Standard deviation | 0.104 |

The whole distribution is shifted slightly negative (median -0.029). A shift like that is what you would expect between the middle of May and the end of
August in a northern city, but the report cannot say how much of it is season; see below.

```python
# check: the numbers in the report fit each other
before, after = 0.287, 0.249
print(f"mean change {after - before:+.3f} (report: -0.038)")
p5, p25, threshold = -0.216, -0.066, -0.20
print(f"p5 {p5} < {threshold} < p25 {p25}: between 5 and 25 % must be flagged, and 5.8 % is")
```

```text
mean change -0.038 (report: -0.038)
p5 -0.216 < -0.2 < p25 -0.066: between 5 and 25 % must be flagged, and 5.8 % is
```

The mean change is exactly the difference of the two means because both are over the same valid pixels.

## What the 5.8 % is, and is not

| It is | It is not |
|---|---|
| The share of **valid** pixels (cloud-free on both dates) whose NDVI fell by more than 0.2. | A share of the box (that is 2.6 %). |
| A flag on *any* vegetation that browned or was cleared. | A measure of crop stress. There are no fields here. |
| Over pixels of every kind: grass, trees, roofs, water. | Limited to vegetation: nothing masks the buildings or the river. |
| A comparison of 20 May with 23 August. | A comparison of like with like. |

```python
# check: how far apart the two scenes are
from datetime import date
print((date(2024, 8, 23) - date(2024, 5, 20)).days, "days between the scenes")
```

```text
95 days between the scenes
```

The report's own limitation says it: "Choose two dates in the same phenological stage; seasonal change otherwise dominates the signal."
Here the scenes are 95 days apart, spring leaf-out to late summer, so part of the drop may be season. In parks, which are the
vegetation that is there to flag, **mown, dry or drought-stressed grass and a tree that has lost leaf look the same in this layer as
a cleared site**. The report says so as well: "Harvest, mowing, phenology and clearing look the same."

## Getting something you can use from it

The flagged raster has all the pixels. Land cover can tell you which *kind* of pixel each flag is in. Take the `land_cover.tif` from
the [first London run](london-landcover-habitat.md) and count the flagged pixels per class:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: flagged_by_land_cover
```

```python
by_class = flagged_by_land_cover("out/crop/crop_health_anomaly.tif", "out/lc/land_cover.tif")
print(by_class)    # {10: ..., 30: ..., 50: ..., 80: ...}: tree cover, grassland, built-up, water
```

The keys are the WorldCover values: 10 tree cover, 30 grassland, 50 built-up, 80 permanent water. What you will learn is how much of
the 5.8 % is on ground that could be vegetation at all, and how much is on roofs and water where an NDVI change means something
else. This page has not run it on the real outputs of this box, because only the PDF reports were available; the function is tested on synthetic
rasters on different grids. For a real study you would then keep the tree and grass flags, and repeat the run with two dates in the same
season.

## Things to be careful about

- **44 % valid.** More than half the box has no answer, and which half depends on where the clouds were on two days. Percentages here
  are over a patchwork.
- **Two scenes.** One day each. A hazy day or a different sun angle shifts NDVI everywhere.
- **No mask for water or buildings.** Their NDVI changes are in the numbers.
- **Resolution.** At 10 m a typical flagged patch is ten pixels. Do not read single patches as events.
- **The threshold is yours.** 0.2 flags 5.8 %; the default 0.15 would flag more than that.

## Change something

| To | Use | Note |
|---|---|---|
| Same season | `--date-before 2023-07 --date-after 2024-07` | Same month, different years: removes most of the seasonal part. |
| A stricter flag | `--param anomaly_threshold=0.3` | Only severe losses. |
| Only vegetation | Mask the result with land cover | See the section above; or restrict to your own boundaries (fields, parks). |

## The files it wrote

| File | What it holds |
|---|---|
| `crop_health_anomaly.tif` | 0 not flagged, 1 NDVI dropped, 255 no data |
| `ndvi_change.tif` | After minus before, continuous; NaN is no data |
| `deliverables/crop_health_report.pdf`, `.xlsx`, `.docx` | The report |
| `deliverables/maps/*.png` | The flag map, the change map, the chart and the histogram |
| `deliverables/vectors/crop_health.gpkg` | Layer `ndvi_drop`: 436 polygons, 49.6 ha, and the `aoi` outline |
| `run_summary.json` | The record, with the two scenes and every parameter |

## Related

- [`crop_health`](../pipelines/catalog/crop_health.md): arguments, statistics and failure messages.
- [`crop_monitoring`](../pipelines/catalog/crop_monitoring.md): crop-type mapping with your own trained model.
- [Land cover and habitat](london-landcover-habitat.md) and [buildings and roads](london-buildings-roads.md): the same box.
- [Radiometry](../data/radiometry.md) and [reading results](../pipelines/results.md).
