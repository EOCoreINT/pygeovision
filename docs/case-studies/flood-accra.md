# Flood extent on the Accra coast: radar, optical and terrain

A flood map is only as useful as your ability to say *how much of it you believe*. This run asks one question of a 1,470 km² box on
the coast of Ghana, **where is there newly flooded ground, and how many independent sensors agree?**, and answers it with three kinds
of evidence that are kept apart instead of being blended: Sentinel-1 radar, Sentinel-2 optical imagery and a terrain model. The
answer is small, 81.4 ha, which is exactly why the run is worth reading closely: it shows what each sensor contributed, what the
cloud hid, and what the terrain filter took out.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`multisensor_flood_mapping_report.pdf`, PyGeoVision 2.1.8,
  generated 2026-10-11 10:28 UTC). The arithmetic between them, and the grid and the terrain limit, are re-derived by
  `tests/test_case_studies.py`.
- **Commands:** parsed by the real command line in the same test, and the Python form must ask for the same thing.
- **Not re-run:** the imagery download and the analysis were not repeated for the documentation (no network there).
- **Not an accuracy assessment:** the report says "there is no ground truth here, so accuracy is not assessed". Nothing on this page
  claims the 81.4 ha is what actually flooded.
```

## The result at a glance

| Figure | Value | Where it comes from |
|---|---:|---|
| Flooded area (classes 1 to 3) | **81.4 ha** | confirmed + radar only + optical only |
| Flooded share of land | **0.1 %** | 81.4 ha of 89,440.8 ha of land |
| Confirmed by radar **and** optical | 4.2 ha | lower bound |
| Radar only | 18.9 ha | optical had no clear view |
| Optical only | 58.2 ha | radar did not confirm |
| Disputed | 3.4 ha | radar flooded, a clear optical view says dry |
| Upper bound (flooded + disputed) | 84.8 ha | |
| Already water in the reference window | 59,501.0 ha | the sea, lagoons and rivers; excluded from the flooded area and from every percentage |
| Removed because it lies more than 15 m above the nearest drainage | 9.0 ha | the terrain filter |
| Share of land with a clear optical view | 18.1 % | why most of the box could only be judged by radar |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -0.3500, 5.4500, 0.0500, 5.7500 |
| Size | 1,470.23 km² (raster extent 1,489.42 km²) |
| Output grid | 2,227 × 1,672 pixels, EPSG:32630, 20 m |
| Reference window requested | 2026-06-01 to 2026-06-20 |
| Flood window requested | 2026-06-27 to 2026-07-06 |
| Scenes actually used | dates 2026-06-02 to 06-09 and 2026-06-27 to 07-03 (the span the report prints as the period) |
| Radar | Sentinel-1 RTC, VV, ascending, relative orbit 147: 3 scenes in the reference window, 2 in the flood window |
| Optical | Sentinel-2 L2A: 10 scenes in the reference window, 6 in the flood window |
| Terrain | Copernicus 30 m DEM, height above the nearest drainage (HAND) |

The box covers the coast around Accra, so the south of it is the Gulf of Guinea. That is why 39.9 % of the output is "water in the
reference window": it is the sea, and it is kept out of every percentage below.

## Run it

```bash
pygeovision channel multisensor_flood_mapping \
    --bbox -0.35 5.45 0.05 5.75 \
    --date-before 2026-06-01/2026-06-20 \
    --date-after 2026-06-27/2026-07-06 \
    --output out/ghana_multisensor_new
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "multisensor_flood_mapping",
    bbox=(-0.35, 5.45, 0.05, 5.75),
    date_before="2026-06-01/2026-06-20",
    date_after="2026-06-27/2026-07-06",
    output_dir="out/ghana_multisensor_new",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["flooded_area_ha"], result.stats["excluded_by_hand_ha"])
```

Nothing else was set: every other argument kept its default, which is why the report's "Pipeline" table lists only the box and the
two windows.

| You give | Meaning here |
|---|---|
| `--bbox` | West, south, east, north in degrees. |
| `--date-before` | The **dry reference** window. A range, so that it contains at least one Sentinel-1 pass of the chosen orbit. |
| `--date-after` | The **flood** window. A single day usually holds no pass, so give a week or two. |
| `--output` | The folder for the rasters, `deliverables/` and `run_summary.json`. |

The windows are wider than the scenes used on purpose. The pipeline searches the window, keeps one orbit track, and prints the span
of the scenes it really used.

## What the pipeline did

```{figure} ../_static/img/flow-case-flood.svg
:alt: Multisensor flood mapping: choose a dry reference window and a flood window, compare Sentinel-1 radar, compute Sentinel-2 MNDWI, class each pixel by what each sensor says, remove ground high above the nearest drainage, write the maps and reports
:width: 100%
:class: pgv-figure

The six stages of `multisensor_flood_mapping`. The dashed card is what happened in the other flood run, on the [Nepal page](flood-nepal.md).
```

1. **Radar.** The per-pixel median backscatter in decibels of each window is compared. Water is a pixel below the radar water
   threshold, and a flooded pixel must also be at least 3 dB darker than in the dry window. The threshold was **-13.8 dB**, found
   by the pipeline, not assumed: it chose the value that makes radar water in the dry window best match optical water in the same
   window (F1 0.99). The conventional default is -15 dB. The two windows also differ slightly in overall brightness, so the pipeline
   removed an average offset of **-0.42 dB** before taking the difference.
2. **Optical.** MNDWI = (green - SWIR1) / (green + SWIR1) is taken per scene with cloud, shadow and cirrus masked by the product's
   Scene Classification Layer, then the per-pixel median of the clear observations is taken for each window. Water is MNDWI above 0.
   A cloudy pixel is *unknown*, never "dry".
3. **Agree.** Each pixel is classed by what each sensor says: both, radar only (the optical view was not clear), optical only, or
   disputed (radar says flooded, a clear optical view says dry). Water that was already there in the reference window is class 5.
4. **Terrain.** A flooded pixel more than 15 m above the nearest drainage is removed (9.0 ha here). Patches under 0.1 ha are removed
   too (15.3 ha).
5. **Deliver.** Four rasters, three reports, maps, vectors and `run_summary.json` (see [the files](#the-files-it-wrote)).

## The result

```{figure} ../_static/img/case-studies/flood-accra-map.webp
:alt: Map of flood extent by sensor agreement over the Accra coast. Almost all of the output is light grey for not flooded or blue for water in the reference window; the flooded classes are a few dots.
:width: 100%
:class: pgv-figure

The report's map. The sea fills the south-east; the flooded classes are too small to see at this scale, which is itself the result.
```

| Class | Value | Pixels | Area (ha) | Share of valid area |
|---|---:|---:|---:|---:|
| Not flooded | 0 | 2,233,899 | 89,356.0 | 60.0 % |
| Flooded: radar + optical | 1 | 105 | 4.2 | 0.0 % |
| Flooded: radar only | 2 | 473 | 18.9 | 0.0 % |
| Flooded: optical only | 3 | 1,456 | 58.2 | 0.0 % |
| Disputed | 4 | 86 | 3.4 | 0.0 % |
| Water in reference window | 5 | 1,487,525 | 59,501.0 | 39.9 % |

```{figure} ../_static/img/case-studies/flood-accra-classes.webp
:alt: Bar chart of area by class: optical only 58.2 hectares, radar only 18.9, radar and optical 4.2, disputed 3.4
:width: 100%
:class: pgv-figure

Area by class. Optical only is the largest of the flooded classes.
```

### Check the numbers yourself

Every figure above follows from the class pixel counts and the pixel size (20 m is 0.04 ha). This function re-derives the headline
numbers from a report's class counts, so you can run the same check on any of your own runs:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: flood_summary
```

```python
# check: re-derive the Accra headline figures from the class pixel counts
pixels = dict(not_flooded=2233899, radar_optical=105, radar_only=473, optical_only=1456, disputed=86, reference_water=1487525)
for name, value in flood_summary(pixels, pixel_m=20).items():
    print(f"{name:34s} {value:10.2f}")
```

```text
flooded_ha                              81.36
flooded_upper_ha                        84.80
land_ha                              89440.76
flooded_share_of_land_pct                0.09
optical_only_share_of_flooded_pct       71.58
confirmed_share_of_flooded_pct           5.16
```

81.36 ha rounds to the report's 81.4 ha, 84.80 to its upper bound of 84.8 ha, and 0.09 % to "0.1 %". The same function says something
the report does not print: **72 % of the flooded area is optical only and 5 % is confirmed by both**.

## How to read it

**The sea is not the story.** 59,501 ha of the box was already water, and the pipeline removed it before computing any share.
Without that, "flooded share" would be diluted by the ocean. Look for the line "water in the reference window" before you quote a
percentage from any flood report.

**Optical sees more, radar confirms less.** The optical map flags 62.4 ha as newly wet (confirmed plus optical only). Radar flags
26.6 ha (confirmed, radar only and disputed). Only 4.2 ha is flagged by both. The classes are not rivals; they carry different
promises:

| Class | What it promises | What it does not promise |
|---|---|---|
| Radar + optical | Two independent sensors agree. Trust this most. | That it is the whole flood. |
| Radar only | Radar saw a new dark patch and optical could not check (cloud). | That it is real; nothing contradicted it. |
| Optical only | A clear optical view saw new water that radar did not confirm. | That radar is wrong. Floodwater among buildings or under vegetation is bright to radar. |
| Disputed | Radar flooded, a clear optical view says dry. | Which sensor is right. Check these on the ground or on high-resolution imagery. |

**Where both sensors could look, they agreed about half the time.** Of the 191 pixels where radar saw flooding *and* optical had a
clear view, 105 (55 %) were confirmed and 86 (45 %) were disputed. That is a small sample and it says nothing about accuracy, but it
is a hint that a radar-only pixel in the cloudy 82 % of the box deserves less trust than the class name suggests.

**The F1 of 0.99 is not about floods.** It scores radar water against optical water in the *dry* window, and most of that water is
the sea, which both sensors see easily. It tells you the calibration step worked. It says nothing about how well shallow, short-lived
floodwater is mapped.

**Why optical had so little to say.** Only 18.1 % of the land had a clear optical view in the flood window, and the report's MNDWI
map is blank over most of the box. In the wet season the cloudiest dates are often the flood dates (the report lists this among its
limitations), and radar is the only one of the two sensors that sees through cloud. That is the reason to run this pipeline instead
of an optical one in a climate like this.

### The other layers

```{figure} ../_static/img/case-studies/flood-accra-radar-change.webp
:alt: Map of radar backscatter change between the flood window and the reference window in decibels, mostly pale with scattered darker and brighter specks
:width: 100%
:class: pgv-figure

Radar backscatter change, flood minus reference (dB). Most of the box is near zero.
```

The change layer is the radar evidence. Its median is +0.10 dB and the middle 90 % of pixels lie between -1.9 and +1.9 dB (standard
deviation 1.37 dB). A flooded pixel must be at least 3 dB darker, so the flood is in the far left tail of this distribution, not in
its body:

```{figure} ../_static/img/case-studies/flood-accra-radar-change-hist.webp
:alt: Histogram of radar backscatter change centred near zero dB, spanning about minus four to plus four decibels
:width: 100%
:class: pgv-figure
```

```{figure} ../_static/img/case-studies/flood-accra-mndwi.webp
:alt: Map of optical MNDWI in the flood window; the sea is strongly positive, clouded areas are blank, and land is near or below zero
:width: 100%
:class: pgv-figure

Optical MNDWI in the flood window. The blank areas have no clear view.
```

```{figure} ../_static/img/case-studies/flood-accra-hand.webp
:alt: Map of height above the nearest drainage in metres, showing a branching valley network of low values among higher ground
:width: 100%
:class: pgv-figure

Height above the nearest drainage (m). The 15 m filter keeps the valley floors and removes the ground above them.
```

HAND has a median of 9.4 m here, with a quarter of the land under 3.6 m and a quarter over 17.2 m. A floodplain is low by
definition, so a "flood" 40 m above the nearest channel is usually something else (a shadow, a dark roof, wet bare soil). The
filter removed 9.0 ha of such candidates.

## Things to be careful about

- **Urban floods are under-mapped.** The report's own limitation: water between buildings and under vegetation is bright to radar and
  hard to see in optical imagery. Accra is a city; use local reports or high-resolution imagery for the built-up part.
- **A peak between passes is missed.** Water that receded before the acquisitions is not seen.
- **The optical classes describe the clear days only.** "Confirmed" and "optical only" refer to the 18.1 % of land that was
  visible, so the optical total is a floor, not an estimate for the whole box.
- **The thresholds are conventional.** MNDWI 0 and the 3 dB change are defaults. The radar threshold was calibrated on this scene; the
  others were not.
- **No ground truth.** Compare with a flood map from an authority, drone imagery or field reports before you use a hectare figure.

## Change something

The [pipeline page](../pipelines/catalog/multisensor_flood_mapping.md) lists every argument. These are the ones this run suggests:

| To | Use | What changes |
|---|---|---|
| Be stricter about terrain | `--param max_hand_m=10` | Floods more than 10 m above a channel are removed (the run used 15). |
| Keep everything, whatever the height | `--param max_hand_m=0` | Switches the terrain filter off. |
| Ignore the calibration | `--param water_db=-15` | Forces the conventional radar threshold; `otsu` uses the scene's own histogram. |
| Drop small speckles | `--param min_patch_ha=0.5` | Removes patches under half a hectare (the run used 0.1 ha). |
| See radar alone | `--param use_optical=false` | Radar plus terrain, no optical confirmation. |
| Look closer | `--param resolution_m=10` | Four times the cells. **The terrain stage then refuses the grid**: see below. |

Check a resolution against the terrain limit before you run, with the same grid the pipeline builds:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: hand_fits
```

```python
# check: does the Accra box fit the terrain stage at other resolutions?
bbox = (-0.35, 5.45, 0.05, 5.75)
for resolution in (20, 15, 10):
    cells, fits = hand_fits(bbox, resolution)
    print(f"{resolution:2d} m  {cells:>10,} cells  terrain {'runs' if fits else 'is skipped'}")
```

```text
20 m   3,723,544 cells  terrain runs
15 m   6,617,901 cells  terrain is skipped
10 m  14,886,379 cells  terrain is skipped
```

The limit is 6,000,000 cells. Going finer than 20 m on this box would silently trade the terrain filter for resolution, and the
report would say so in its data notes. Either keep 20 m, or cut the box. The [Nepal page](flood-nepal.md) is the run where this
happened.

## The files it wrote

| File | What it holds |
|---|---|
| `flood_extent.tif` | Class 0 to 5 as above, 255 for no data |
| `change_db.tif` | Radar change in dB |
| `mndwi_after.tif` | Optical MNDWI in the flood window |
| `hand_m.tif` | Height above the nearest drainage, in metres |
| `deliverables/multisensor_flood_mapping_report.pdf`, `.xlsx`, `.docx` | The complete report in three formats |
| `deliverables/maps/*.png` | The map, three layer maps, the chart and the histogram, for slides |
| `deliverables/vectors/multisensor_flood_mapping.gpkg` | The flooded polygons (layer `flood_extent`) and the `aoi` outline |
| `deliverables/vectors/geojson/*.geojson` | The same in WGS 84 GeoJSON |
| `run_summary.json` | Everything above in one machine-readable record |

The vector layer has 331 features with a total of 84.6 ha: classes 1 to 4 (the 84.8 ha upper bound, measured in an equal-area
projection). Water in the reference window is not in it. Summarise it by class:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: vector_area_by_class
```

```python
areas = vector_area_by_class(
    "out/ghana_multisensor_new/deliverables/vectors/multisensor_flood_mapping.gpkg", "flood_extent")
print(areas)   # area_ha per class_name
```

## Related

- [Nepal flood run](flood-nepal.md): the same pipeline where terrain was skipped and optical dominated.
- [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md): every argument, output and failure.
- [Drainage analysis in Tyrol](drainage-tyrol.md): how HAND is built.
- [Reading results](../pipelines/results.md) and [the report](../reporting/contents.md).
- [Radar data](../data/sar.md) and [choosing scenes](../data/scene-selection.md).
