# Flood extent in central Nepal: a run that needs a second look

The [Accra run](flood-accra.md) shows the flood pipeline at its best: three sensors, a calibrated threshold, a terrain filter. This
run uses the same pipeline on a box more than three times larger, and the report shows what goes wrong when the conditions are
harder. It reports 451.7 ha as flooded, but **90 % of that is "optical only", only 3 % is confirmed by both sensors, and the terrain
filter did not run**. The value of the page is learning to read those three facts *before* quoting the hectares, and knowing the one
line that would put the terrain filter back.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`multisensor_flood_mapping_report.pdf`, PyGeoVision 2.1.8,
  generated 2026-10-11 10:02 UTC). Arithmetic, grid sizes and the terrain limit are re-derived by `tests/test_case_studies.py`.
- **Not re-run:** nothing was downloaded or recomputed for the documentation, and the rerun suggested below has **not** been
  tried. The page says what the grid arithmetic allows, not what the rerun will find.
- **Not an accuracy assessment:** the report states that there is no ground truth. The explanations offered for the optical-only
  share are hypotheses to test, not findings.
```

## The result at a glance

| Figure | Value |
|---|---:|
| Flooded area (classes 1 to 3) | **451.7 ha** |
| Confirmed by radar and optical | 13.2 ha |
| Radar only | 31.5 ha |
| Optical only | **407.1 ha** (90.1 % of the flooded area) |
| Disputed | 14.4 ha |
| Upper bound (flooded + disputed) | 466.1 ha |
| Already water in the reference window | 14,503.0 ha (2.7 % of the box) |
| Removed by terrain (HAND) | **0.0 ha, because the terrain stage did not run** |
| Share with a clear optical view | 19.8 % |
| Radar water threshold | -15.0 dB, "conventional default" |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | 84.8000, 27.7000, 85.6000, 28.3000 (central Nepal) |
| Size | 5,232.20 km² (raster extent 5,391.15 km²) |
| Output grid | 3,984 × 3,383 = **13,477,872 pixels**, EPSG:32645, 20 m |
| Reference window requested | 2026-08-10 to 2026-08-24 |
| Flood window requested | 2026-08-26 to 2026-08-28 |
| Period in the report | 2026-08-16 to 2026-08-28 (the span of scenes used) |
| Radar | Sentinel-1 RTC, VV, ascending, relative orbit 85: 2 scenes in each window |
| Optical | Sentinel-2 L2A: 10 scenes in the reference window, **1 in the flood window** |
| Terrain | **not applied** |

## Run it

```bash
pygeovision channel multisensor_flood_mapping \
    --bbox 84.8 27.7 85.6 28.3 \
    --date-before 2026-08-10/2026-08-24 \
    --date-after 2026-08-26/2026-08-28 \
    --output out/nepal_multisensor
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "multisensor_flood_mapping",
    bbox=(84.8, 27.7, 85.6, 28.3),
    date_before="2026-08-10/2026-08-24",
    date_after="2026-08-26/2026-08-28",
    output_dir="out/nepal_multisensor",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["flooded_area_ha"], result.stats["hand_applied"])
```

The report records no other argument, so defaults applied. The three facts that make this run different from Accra all come from
those defaults meeting this box and this weather. The flood window is only three days wide, which is why it holds a single optical
scene.

## Three things the report tells you, in the order they matter

### 1. The terrain filter did not run

The executive summary says it in one line: *"terrain (HAND) not applied: the area has 13.5 million cells; the limit for the flow
analysis is 6.0 million. Use a smaller bounding box or a coarser resolution_m."* The run did not fail. It carried on without terrain
and said so in the "Data notes" and in the list of data used (sentinel-1 and sentinel-2, with no `copernicus-dem`). Everywhere in the
report, "Removed by terrain (HAND)" reads 0.0 ha, and "Hand applied" reads **no**.

That is a real loss in a box whose northern edge reaches the high Himalaya (judging by its coordinates). The terrain filter exists to
remove "floods" on ground that is physically too high to flood. Accra's run removed 9.0 ha with it. On ground with this much relief
the filter would normally have more to remove than in a coastal box, but the report cannot say how much: that is the number you
do not have.

The cause is plain arithmetic, and you can check it before spending a download. The grid is the one `make_grid` builds for the box,
and the terrain stage accepts at most 6,000,000 cells:

```python
# check: how many cells does the Nepal box give, and which splits let the terrain stage run?
west, south, east, north = 84.8, 27.7, 85.6, 28.3
bbox = (west, south, east, north)
for resolution in (20, 30):
    cells, fits = hand_fits(bbox, resolution)
    print(f"whole box at {resolution} m: {cells:>10,} cells, terrain {'runs' if fits else 'is skipped'}")
mid_lon, mid_lat = (west + east) / 2, (south + north) / 2
for name, sub in {"south-west": (west, south, mid_lon, mid_lat), "south-east": (mid_lon, south, east, mid_lat),
                  "north-west": (west, mid_lat, mid_lon, north), "north-east": (mid_lon, mid_lat, east, north)}.items():
    cells, fits = hand_fits(sub, 20)
    print(f"{name:10s} at 20 m: {cells:>10,} cells, terrain {'runs' if fits else 'is skipped'}")
```

```text
whole box at 20 m: 13,477,872 cells, terrain is skipped
whole box at 30 m:  5,991,936 cells, terrain runs
south-west at 20 m:  3,386,610 cells, terrain runs
south-east at 20 m:  3,366,177 cells, terrain runs
north-west at 20 m:  3,376,440 cells, terrain runs
north-east at 20 m:  3,356,043 cells, terrain runs
```

The first line reproduces the report's 13.5 million. There are two ways to bring the terrain filter back:

| Way | Command | Trade-off |
|---|---|---|
| A coarser grid | `--param resolution_m=30` | One run, but 30 m pixels (0.09 ha each) blur small floods. 5,991,936 cells is only 8,064 under the limit. |
| Four smaller boxes | one run per quarter, 20 m each | Four runs, full resolution, and four reports to merge. Flow is cut at each box edge, so streams near a seam read short. |

```bash
pygeovision channel multisensor_flood_mapping \
    --bbox 84.8 27.7 85.6 28.3 \
    --date-before 2026-08-10/2026-08-24 \
    --date-after 2026-08-26/2026-08-28 \
    --param resolution_m=30 \
    --output out/nepal_multisensor_30m
```

This command is the grid-checked rerun. It has not been executed; what is verified is that it builds a grid the terrain stage will
accept.

```python
result = client.pipeline(
    "multisensor_flood_mapping",
    bbox=(84.8, 27.7, 85.6, 28.3),
    date_before="2026-08-10/2026-08-24",
    date_after="2026-08-26/2026-08-28",
    resolution_m=30,
    output_dir="out/nepal_multisensor_30m",
)
```

### 2. The radar threshold is a default, not a calibration

The Accra run found its threshold (-13.8 dB, F1 0.99) by matching radar water to optical water in the dry window. This one reports
**-15.0 dB, source "conventional default"**, and no radar-optical agreement figure. The report does not say why. The code takes that
branch when Sentinel-2 gave no usable view of the reference window, or fewer than 200 pixels of water or of land to calibrate on. The
report also lists the scene's own Otsu threshold (-7.1 dB) next to the default.

A default is not wrong, but it is not tuned to this scene, and there is no agreement figure to tell you how close it is.

### 3. Almost all the flooding is "optical only"

| | Accra | Nepal |
|---|---:|---:|
| Box | 1,470 km² | 5,232 km² |
| Grid cells | 3.7 million | 13.5 million |
| Terrain filter applied | yes (15 m) | **no** |
| Radar threshold | -13.8 dB, calibrated | -15.0 dB, default |
| Optical scenes in the flood window | 6 | **1** |
| Clear optical view | 18.1 % | 19.8 % |
| Flooded area | 81.4 ha | 451.7 ha |
| Optical only, share of flooded | 71.6 % | **90.1 %** |
| Confirmed by both, share of flooded | 5.2 % | **2.9 %** |
| Radar flagged (confirmed + radar only + disputed) | 26.6 ha | 59.0 ha |
| Optical flagged (confirmed + optical only) | 62.4 ha | 420.2 ha |
| Of radar floods optical could check, share confirmed | 55 % | 47.8 % |

Reproduce the Nepal column's headline from the class pixel counts:

```python
# check: re-derive the Nepal headline figures from the class pixel counts
pixels = dict(not_flooded=13103644, radar_optical=329, radar_only=787, optical_only=10177, disputed=360, reference_water=362575)
for name, value in flood_summary(pixels, pixel_m=20).items():
    print(f"{name:34s} {value:10.2f}")
```

```text
flooded_ha                             451.72
flooded_upper_ha                       466.12
land_ha                             524611.88
flooded_share_of_land_pct                0.09
optical_only_share_of_flooded_pct       90.12
confirmed_share_of_flooded_pct           2.91
```

Radar and optical disagree by a factor of seven about how much is wet (59.0 ha against 420.2 ha), and the sensors that disagree are
not equally good at the same things. A class 3 pixel needs a clear optical view that says water now, a reference view that said
"not water", and *no* radar confirmation. So it lumps two different situations that the report does not separate:

- floodwater the radar could not see (bright to radar under vegetation or among buildings), where optical is right; and
- something that is not floodwater but has a high MNDWI in a single scene: a cloud shadow, a wet field, snow or ice.

This run gives you the conditions that make the second more likely: **one** optical scene in the flood window, against a median of
ten in the reference window; no terrain filter; and, in the north of the box, high relief. That is a hypothesis, not a finding. The way to test it is
to rerun with the terrain filter and see how much of class 3 sits high above a channel, and to look at the class 3 patches on
high-resolution imagery.

## The result

```{figure} ../_static/img/case-studies/flood-nepal-map.webp
:alt: Map of flood extent over central Nepal; rivers and lakes already present in the reference window are blue, and the flooded classes are specks too small to read at this scale
:width: 100%
:class: pgv-figure

The report's map. Blue is water already present in the reference window (the river network and lakes at the top right); the flooded classes are specks too small to read at this scale.
```

| Class | Value | Pixels | Area (ha) | Share of valid area |
|---|---:|---:|---:|---:|
| Not flooded | 0 | 13,103,644 | 524,145.8 | 97.2 % |
| Flooded: radar + optical | 1 | 329 | 13.2 | 0.0 % |
| Flooded: radar only | 2 | 787 | 31.5 | 0.0 % |
| Flooded: optical only | 3 | 10,177 | 407.1 | 0.1 % |
| Disputed | 4 | 360 | 14.4 | 0.0 % |
| Water in reference window | 5 | 362,575 | 14,503.0 | 2.7 % |

```{figure} ../_static/img/case-studies/flood-nepal-classes.webp
:alt: Bar chart of area by class: water in the reference window 14,503 hectares dwarfs optical only 407, radar only 31.5, disputed 14.4 and confirmed 13.2
:width: 100%
:class: pgv-figure

Area by class. The scale is set by the reference water, so the flooded classes are slivers.
```

### What the two sensors saw

```{figure} ../_static/img/case-studies/flood-nepal-mndwi.webp
:alt: Map of optical MNDWI in the flood window with data only in the south-east quarter; everywhere else is blank
:width: 100%
:class: pgv-figure

Optical MNDWI in the flood window: one scene, and it covers only the south-east of the box.
```

The optical evidence exists for about a fifth of the box. Its values are mostly strongly negative (median -0.443; 95th percentile
-0.161), which is what land looks like; only the tail above 0 is water. Of 2,698,840 clear pixels, 10,177 are class 3: 0.38 %.

```{figure} ../_static/img/case-studies/flood-nepal-radar-change.webp
:alt: Map of radar backscatter change in decibels over central Nepal, pale and near zero almost everywhere, with a thin dark-red line along the main river in the centre-left and a few dark spots at the top right
:width: 100%
:class: pgv-figure

Radar change, flood minus reference (dB), covering the whole box. The dark-red line along the main river is where radar saw new dark water.
```

Radar covers the whole box, and its change layer is close to zero almost everywhere (mean -0.044 dB, middle 90 % between -2.0 and
+1.9 dB). The exceptions are a thin dark-red line along the main river in the centre-left and a few dark spots at the top right: that
is the radar's flood signal. A flood must be at least 3 dB darker, so the radar tail it counts is small, and it is concentrated along
a river. That pattern is what a river leaving its banks looks like, but the report cannot tell overbank flooding from a river that is
simply wider than in the reference window. Radar saw little because there was little *it* could call new water, not because it could
not look.

## Smaller things worth knowing

- **127.8 ha of small patches were removed** (patches under 0.1 ha). That is more than a quarter of the 451.7 ha that remained,
  which says the raw candidate map was speckled.
- **The vector layer includes the reference water.** Its 14,023 features total 14,971.2 ha, which is classes 1 to 5
  (13.2 + 31.5 + 407.1 + 14.4 + 14,503.0 = 14,969.2 ha, measured again in an equal-area projection). The current code writes only
  classes 1 to 4 to this layer, as the Accra report's 84.6 ha shows, so a rerun will give a layer of about 466 ha. If you already
  use this run's GeoPackage, filter on `class_name` and drop *Water in reference window*.
- **The classes are conventions, not a score.** There is no per-pixel confidence. The disputed class is the only place the report
  flags disagreement directly.

## What to do with this run

| Do | Do not |
|---|---|
| Quote the **13.2 ha** confirmed and the **466.1 ha** upper bound together, and say the terrain filter did not run. | Quote 451.7 ha as "the flood". |
| Rerun with terrain (30 m or four quarters) and compare `excluded_by_hand_ha` and the optical-only share. | Trust class 3 on this run before that rerun. |
| Widen the flood window if you need more clear optical scenes, knowing that a wider window lets the flood recede inside it. | Treat the 3-day window as a long observation. |
| Check class 3 and class 4 patches on high-resolution imagery or against a local flood map. | Read the -15 dB default as calibrated. |

## The files it wrote

The same set as the [Accra run](flood-accra.md#the-files-it-wrote), except that there is no `hand_m.tif` (no terrain was computed) and
the vector layer holds classes 1 to 5.

## Related

- [Accra flood run](flood-accra.md): the same pipeline with terrain and a calibrated threshold.
- [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md): every argument, `max_hand_m` and `resolution_m`
  among them.
- [Drainage analysis in Tyrol](drainage-tyrol.md): what the missing terrain layer looks like, and how steep ground shows in HAND.
- [Troubleshooting](../getting-started/troubleshooting.md).
