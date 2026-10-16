# Water-quality indicators on the upper Chesapeake Bay: a report to question

`water_quality` finds water in one Sentinel-2 scene and, inside it, computes two **relative** indicators: NDCI, a proxy for
chlorophyll-a, and red reflectance, a proxy for turbidity. It is cheap, needs no account and is a useful first look at where algae or
sediment might be. This run covers a 1,539 km² box around the upper Chesapeake Bay and is included for a different reason than the
others: **the report's own figures let you catch that something is off**, and learning to do that is worth more than the map. Four
numbers in the report do not fit together, and the page shows which, why they matter, and what to run to find out more.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`water_quality_report.pdf`, PyGeoVision 2.1.8, generated
  2026-10-10 16:28 UTC), and checked for consistency in `tests/test_case_studies.py`.
- **What was reproduced:** the *mechanism* described below, on the real `water_quality` pipeline with a small scene whose answer is
  known. Whether this particular scene has that problem **cannot be decided from the report**; the page gives the commands that
  decide it.
- **Not an accuracy assessment:** NDCI and red reflectance are relative indicators; converting them to chlorophyll-a (µg/l) or
  turbidity (NTU) needs local water samples, which this run did not use.
```

## The result at a glance

| Figure | Value |
|---|---:|
| Water share of valid pixels (NDWI above 0) | **2.1 %** |
| Mean NDCI in water | 0.249 |
| Median NDCI in water | 0.217 |
| Mean red reflectance in water | 0.013 |
| Valid (cloud-free) share of the box | **40.7 %** |
| Water pixels with a red reflectance | 127,989 |
| Water pixels with an NDCI | **66,612** |

The last two lines are the story of this page.

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -76.6000, 38.8000, -76.2000, 39.2000 (the upper Chesapeake Bay, Maryland, by its coordinates) |
| Size | 1,538.70 km² |
| Output grid | 3,838 × 3,838 = 14,730,244 pixels, EPSG:4326 (one pixel is about 104.5 m²) |
| Date requested | `2024-06` |
| Scene used | `S2A_18SUJ_20240628_0_L2A` (28 June 2024), covering 100 % of the box |
| Bands | green, red, red-edge 1, NIR |
| Valid share | 40.7 % |

One month is requested and **one scene** is used: the clearest. Everything cloud-masked in that scene is empty, so 59.3 % of the
box has no answer. The report says so in a warning on its first page and on every map.

## Run it

```bash
pygeovision channel water_quality \
    --bbox -76.6 38.8 -76.2 39.2 \
    --date 2024-06 \
    --output out/wq
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "water_quality",
    bbox=(-76.6, 38.8, -76.2, 39.2),
    date="2024-06",
    output_dir="out/wq",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["pct_water"], result.stats["mean_ndci_in_water"], result.stats["valid_fraction"])
```

The pipeline takes only a month and a box.

## What the pipeline did

```{figure} ../_static/img/flow-case-water.svg
:alt: Water quality indicators: one cloud-masked Sentinel-2 scene, water mask from NDWI above zero, NDCI and red reflectance inside the water, terciles of NDCI, report with the valid share
:width: 100%
:class: pgv-figure

Five steps. The dashed card is the one the report makes visible if you read it closely.
```

1. **One scene.** Sentinel-2 L2A surface reflectance, with cloud and shadow pixels left empty.
2. **Water.** NDWI = (green - NIR) / (green + NIR), and water is NDWI above 0.
3. **Indicators.** Inside the water only: NDCI = (red-edge 1 - red) / (red-edge 1 + red) (Mishra and Mishra 2012), and the red
   reflectance itself. Output band 1 is NDCI and band 2 is red reflectance.
4. **Terciles.** The NDCI values are split into a lower, middle and upper third *within this water*.
5. **Report.** With the valid share and the "relative indicators" warning.

NDCI is a ratio, and a ratio is undefined when its denominator is zero. The pipeline does not fill that with a made-up number: where
red and red-edge reflectance add to (nearly) zero, NDCI is left **empty**, while the red reflectance (which is a plain number, zero) is
kept. That is the right thing to do, and it is why the two pixel counts in the table above can differ.

## The result

```{figure} ../_static/img/case-studies/water-quality-ndci.webp
:alt: Map of NDCI inside water over the upper Chesapeake Bay; scattered small water bodies and river reaches are shaded from pale yellow to dark blue, with most of the box blank
:width: 100%
:class: pgv-figure

NDCI inside the water. Most of the box is blank: cloud, or land.
```

| Class | Value | Pixels | Area (ha) | Share of the valid classed area |
|---|---:|---:|---:|---:|
| Lower third of NDCI | 0 | 22,203 | 232.1 | 33.3 % |
| Middle third of NDCI | 1 | 22,205 | 232.1 | 33.3 % |
| Upper third of NDCI | 2 | 22,204 | 232.2 | 33.3 % |

Terciles are equal thirds *by construction*, so the equal bars carry no information; what carries information is **where** each third
is. They describe the 66,612 pixels (696.4 ha) that have an NDCI, not all the water the pipeline found (see below).

| Statistic | NDCI in water (66,612 px) | Red reflectance in water (127,989 px) |
|---|---:|---:|
| Minimum | -1.000 | 0.000 |
| 5th percentile | **-1.000** | 0.000 |
| 25th percentile | -0.026 | 0.000 |
| Median | 0.217 | **0.000** |
| Mean | 0.249 | 0.013 |
| 75th percentile | 0.789 | 0.009 |
| 95th percentile | **1.000** | 0.054 |
| Maximum | 1.000 | 0.868 |
| Standard deviation | 0.582 | 0.044 |

```{figure} ../_static/img/case-studies/water-quality-ndci-hist.webp
:alt: Histogram of NDCI inside water with a tall spike at exactly minus one, a broad low hump between about minus 0.5 and 0.8, and a taller spike at exactly plus one
:width: 100%
:class: pgv-figure

The NDCI histogram. The two spikes at the ends are the pixels the next section is about.
```

## Four numbers that do not fit together

**1. NDCI is pinned at both ends of its range.** The 5th percentile is exactly -1 and the 95th exactly +1, and the histogram has a
tall spike at each end (reading the bars off the chart, roughly 5,000 pixels at -1 and roughly 15,000 at +1, together about
a third of the 66,612). A chlorophyll index in real water spreads out across a range; a spike at the mathematical limits means a
large number of pixels where one of the two reflectances is zero and the other is not (NDCI becomes -1 if red-edge is 0 and red is
positive, +1 if red is 0 and red-edge is positive).

**2. The median red reflectance is 0.000, and the 75th percentile is 0.009.** Water is dark in the red, but it is typically not
zero: turbid estuary water often reads a few hundredths. More than half of the water pixels here have a red reflectance that rounds
to zero.

**3. The two statistics are over different pixels.** The red reflectance is summarised over **127,989** water pixels and the NDCI
over **66,612**. Every water pixel has a red reflectance, so the difference is the water pixels with *no* NDCI:

```python
# check: how many water pixels have a red reflectance but no NDCI?
water_pixels, ndci_pixels = 127_989, 66_612
missing = water_pixels - ndci_pixels
print(f"{missing:,} water pixels have no NDCI, {100 * missing / water_pixels:.1f} % of the water")
print(f"water found: {water_pixels * 696.4 / ndci_pixels:,.0f} ha; classed into terciles: 696.4 ha")
```

```text
61,377 water pixels have no NDCI, 48.0 % of the water
water found: 1,338 ha; classed into terciles: 696.4 ha
```

**Almost half of the water the pipeline found has no NDCI**, and the class table, the terciles and the vector layer
(13,781 features, 696.4 ha) describe only the other half. NDCI is undefined only where red and red-edge reflectance sum to (nearly) zero, so
those 61,377 pixels are ones where both reflectances are at the floor of the product.

**4. Only 2.1 % of the valid pixels are water.** By its coordinates the box reaches into the main stem of the Bay, so a much
larger water share would be expected. The map shows only small, scattered water bodies and river reaches, and no continuous open
water. The report cannot say where the open water went. It may be in the 59.3 % that has no valid data, or the NDWI rule may not
have flagged it; the report does not distinguish, and this page has not looked at the scene.

Taken together, these four point the same way: **a large share of the water pixels in this scene have red and red-edge reflectance
at or near zero**, so NDCI is either empty or pinned at -1 or +1 for them, and the headline means (0.249 and 0.013) are averages over
a population that is not what the words "water quality" suggest.

### Reproduced on the real pipeline

The test suite builds a small scene in which some water pixels have red and red-edge reflectance of exactly zero, and runs the
real `water_quality` code on it. Where both are zero the NDCI band is empty; where only red is zero it is +1; where only red-edge is
zero it is -1; and the red-reflectance band keeps its zero. That is the mechanism, reproduced. It shows that zero reflectance
produces exactly the pattern in this report. **It does not show that this scene has zero-reflectance pixels**, and it says nothing
about why they would be zero (a very dark surface, a limit of the atmospheric correction over water, glint, a clipped value). Only
the data can tell you.

### How to find out

Run this on the `water_quality.tif` the pipeline wrote. It counts the water pixels that have a red reflectance but no NDCI, and the
ones whose red reflectance is exactly zero:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: ndci_undefined
```

```python
result = ndci_undefined("out/wq/water_quality.tif")
print(result)
```

For this run the first three entries would be `water_pixels` 127,989, `ndci_pixels` 66,612 and `missing` 61,377 (the report's numbers).
The one the report does not give is `red_exactly_zero`: if it is large, the problem is in the reflectance values; if it is small,
look at the red-edge band instead. Then:

| Check | Why |
|---|---|
| Open the scene's red and red-edge bands over the water and read a few pixel values | Tells you whether the zeros are in the data or created later. |
| Run the same box for another month (`--date 2024-07`, `2024-08`) | If the spikes shrink, the cause is the scene (haze, glint, a processing difference), not the water. |
| Count the valid water pixels per scene | A 40.7 % valid share from one scene may simply be an unlucky date. |
| Compare with a station, buoy or sample | The only way to learn what 0.249 means in µg/l. |

## What to take from this run

| You can say | You should not say |
|---|---|
| Where, among the pixels that have an NDCI, the indicator is relatively higher or lower (the terciles). | "The Bay has a mean chlorophyll index of 0.249." |
| That 40.7 % of the box was observable on 28 June 2024 and 59.3 % was not. | That the water covers 2.1 % of the area. |
| That the run flags 61,377 water pixels with no NDCI. | That the quality of the Bay was mapped. |

## Things to be careful about

- **Relative, not absolute.** NDCI and red reflectance are indices. Shallow water, bottom reflectance and sun glint also change them.
- **One scene, one day.** A single clear date says nothing about the month it was requested for.
- **Terciles are within this water body.** They are not a quality grade and cannot be compared with another run.
- **Check the pixel counts.** Whenever two statistics in a report come from different pixel counts, find out why before you trust
  either.

## Change something

| To | Do |
|---|---|
| Another month | `--date 2024-07` (the only date argument) |
| A smaller, clearer box | Narrow `--bbox` to one river mouth or a single water body. |
| Cleaner pixels | Mask `water_quality.tif` yourself, for example keeping only pixels where band 2 is above a small positive value. |
| Convert to concentrations | Fit NDCI to your own samples with a regression; PyGeoVision does not do this for you. |

## The files it wrote

| File | What it holds |
|---|---|
| `water_quality.tif` | Band 1 NDCI, band 2 red reflectance, both inside the water only |
| `deliverables/water_quality_report.pdf`, `.xlsx`, `.docx` | The report |
| `deliverables/maps/*.png` | The NDCI map, the turbidity map, the chart and the histogram |
| `deliverables/vectors/water_quality.gpkg` | Layer `ndci_zones`: 13,781 features, 696.4 ha, and the `aoi` outline |
| `run_summary.json` | The machine-readable record |

```{figure} ../_static/img/case-studies/water-quality-red.webp
:alt: Map of red reflectance inside water, almost entirely very pale with a few slightly shaded patches
:width: 100%
:class: pgv-figure

The red-reflectance layer: nearly everything is at the pale end of a 0 to 0.15 scale, consistent with the zero median.
```

## Related

- [`water_quality`](../pipelines/catalog/water_quality.md): arguments, statistics and failure messages.
- [`water_bodies`](../pipelines/catalog/water_bodies.md) and [`surface_water_dynamics`](../pipelines/catalog/surface_water_dynamics.md):
  mapping the water itself.
- [Radiometry](../data/radiometry.md): how reflectance is scaled and offset for Sentinel-2.
- [Reading results](../pipelines/results.md) and [troubleshooting](../getting-started/troubleshooting.md).
