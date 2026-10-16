# Burn severity in northern California: reading a dNBR map critically

`wildfire_severity` turns two Sentinel-2 images into a four-class map of how much vegetation changed, using the burn index dNBR. This
run reports **49,568.1 ha burned** in a 3,321 km² box between October and December 2018. It is the clearest example in these case
studies of a result that is *numerically clean* and still needs a question asked of it, because dNBR measures "NIR fell and SWIR
rose", and a fire is only one of several things that do that. The page walks through the report, explains why two of its own
numbers differ by 20.8 ha, and shows what to look at before calling the map a burn scar.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`wildfire_severity_report.pdf`, PyGeoVision 2.1.8, generated
  2026-10-10 16:41 UTC). The class areas, the shares and the 20.8 ha difference are re-derived in `tests/test_case_studies.py`, and
  the explanation for the difference is reproduced on the real pipeline with a scene whose answer is known.
- **Not re-run:** the imagery was not downloaded again.
- **No event is named.** The report does not say which fire, if any, the dates were chosen for, and neither does this page. It is
  not an accuracy assessment: no reference perimeter was used.
```

## The result at a glance

| Figure | Value |
|---|---:|
| Burned area (low + moderate + high), headline | **49,568.1 ha**, 15.6 % of the valid area |
| Unburned | 269,163.4 ha, 84.4 % |
| Low severity (dNBR 0.10 to 0.27) | 26,781.0 ha, 8.4 % |
| Moderate severity (0.27 to 0.66) | 21,257.5 ha, 6.7 % |
| High severity (0.66 and above) | 1,550.4 ha, 0.5 % |
| Mean dNBR | -0.023 |
| Median dNBR | -0.054 |
| Valid share of the box | 95.9 % |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -122.6000, 39.7000, -121.9000, 40.2000 (northern California) |
| Size | 3,320.96 km² (raster extent 3,322.44 km²) |
| Output grid | 6,703 × 4,788 = 32,093,964 pixels, EPSG:4326 |
| Dates requested | `2018-10` before, `2018-12` after |
| Scenes used | before: `S2B_10TEK_20181020_1_L2A`; after: `S2B_MSIL2A_20181206T185749_R113_T10TEK_20201008T080050` |
| Coverage of the box by each scene | 100.0 % |
| Valid share | 95.9 % (the rest has no data) |

The dates are given as months. PyGeoVision chose the scenes inside each month: the clearest one, here 20 October and 6 December.
That is a gap of about six and a half weeks, from the dry season into the first winter weeks.

## Run it

```bash
pygeovision channel wildfire_severity \
    --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12 \
    --output out/fire
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "wildfire_severity",
    bbox=(-122.6, 39.7, -121.9, 40.2),
    date_before="2018-10",
    date_after="2018-12",
    output_dir="out/fire",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["burned_area_ha"], result.stats["mean_dnbr"])
```

This is the example in the command's own `--help`. The pipeline takes two dates and nothing else: its thresholds are fixed (see
[Change something](#change-something)).

## What the pipeline did

```{figure} ../_static/img/flow-case-fire.svg
:alt: Burn severity: pick a pre-fire and a post-fire date, compute NBR for each, subtract to get dNBR, class it with the USGS thresholds, measure areas over valid pixels, write the maps and reports
:width: 100%
:class: pgv-figure

From two dates to four classes. Cloud-masked pixels are left empty in step 2 and stay empty to the end.
```

1. **Two dates.** One scene each, from the months you give.
2. **NBR twice.** NBR = (NIR - SWIR2) / (NIR + SWIR2), from surface reflectance, for each date. Cloud and shadow pixels are `NaN`.
3. **Subtract.** dNBR = NBR(before) - NBR(after). A burn lowers NIR and raises SWIR2, so NBR drops and dNBR goes **positive**. A
   negative dNBR means NBR *rose*.
4. **Classes.** Simplified USGS thresholds: unburned below 0.10, low 0.10 to 0.27, moderate 0.27 to 0.66, high 0.66 and above.
5. **Measure.** Areas and shares are taken over valid pixels only. A pixel with no data is `nodata`, never "unburned".

## The result

```{figure} ../_static/img/case-studies/wildfire-map.webp
:alt: Map of burn severity in northern California. A long, narrow north-south belt of orange and red patches, with field-sized rectangles on either side and scattered light-yellow specks elsewhere
:width: 100%
:class: pgv-figure

The report's severity map.
```

| Class | Value | Pixels | Area (ha) | Share of valid area | Share of box |
|---|---:|---:|---:|---:|---:|
| Unburned | 0 | 26,003,569 | 269,163.4 | 84.4 % | 81.0 % |
| Low | 1 | 2,585,986 | 26,781.0 | 8.4 % | 8.1 % |
| Moderate | 2 | 2,052,483 | 21,257.5 | 6.7 % | 6.4 % |
| High | 3 | 149,711 | 1,550.4 | 0.5 % | 0.5 % |

The two share columns differ because 4.1 % of the box has no data (the pale wedge on the east edge of the map): "of valid area"
divides by the pixels that could be judged, "of box" by all of them. Prefer the first.

```{figure} ../_static/img/case-studies/wildfire-classes.webp
:alt: Bar chart of area by burn severity class: low 26,781 hectares, moderate 21,257.5, high 1,550.4
:width: 100%
:class: pgv-figure
```

### Why the two burned areas differ by 20.8 ha

The headline says **49,568.1 ha**. Adding the three burned rows of the class table gives 26,781.0 + 21,257.5 + 1,550.4 =
**49,588.9 ha**, and the vector layer totals 49,589.8 ha. They are not the same, and neither is an error. They weight the pixels
differently.

The headline multiplies the number of burned pixels by **one** pixel area. In a box in degrees (EPSG:4326), a pixel is a little
smaller at the north edge than at the south edge, because a degree of longitude shortens as you move poleward. The report's own
numbers show it: 32,093,964 pixels cover 332,243.7 ha, so the average pixel is 0.010352 ha, and 4,788,180 burned pixels times that is
49,568.3 ha. The class table and the vector layer measure each pixel where it is. The burned pixels sit slightly south of the middle
of the box, where pixels are a little larger (the report's rows come to 0.010356 ha per burned pixel against 0.010351 ha for the
unburned), so the table is bigger by 20.8 ha, 0.04 %.

You can reproduce the direction and size of the effect, with no data at all, using the function below. It sums a raster mask row by
row at each row's own latitude:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: rowwise_area_ha
```

```python
# check: how much more ground does the southern half of this box cover than the northern half?
import numpy as np

bbox = (-122.6, 39.7, -121.9, 40.2)
rows, cols = 400, 560
south = np.zeros((rows, cols), bool); south[rows // 2:] = True
north = ~south
s, n = rowwise_area_ha(south, *bbox), rowwise_area_ha(north, *bbox)
print(f"southern half {s:,.1f} ha, northern half {n:,.1f} ha")
print(f"{100 * (s / n - 1):.2f} % more ground in the southern half")
print(f"whole box {rowwise_area_ha(np.ones((rows, cols), bool), *bbox):,.1f} ha")
```

```text
southern half 166,347.0 ha, northern half 165,749.7 ha
0.36 % more ground in the southern half
whole box 332,096.6 ha
```

The test suite goes one step further and runs the real `wildfire_severity` on a synthetic scene whose burned pixels are all in the
south, then all in the north: the headline statistic gives the **same** number both times (it only counts pixels), while the
row-by-row sum is larger for the south and smaller for the north. For a map this size the choice moves the total by 0.04 %. For a
box nearer a pole, or much taller, it would move it more. Quote one method and name it.

## Is it a burn? What to check before you use the map

This is the part the numbers cannot do for you. Four things in the report are worth a closer look.

**The distribution of dNBR is shifted.** The median is -0.054 and the middle half of the pixels lie between -0.130 and +0.033: most
of the box got *greener* between the two dates by this index (NBR rose), and even the 75th percentile is below the 0.10 low-severity
threshold. A burn is a tail above 0.10, but the body of the distribution is not at zero. That is what a change of season does
(leaf-fall, the first rains, a different sun angle), and it is the reason the report advises choosing dates "close to the fire and in
the same season".

```{figure} ../_static/img/case-studies/wildfire-dnbr-hist.webp
:alt: Histogram of dNBR with a peak slightly below zero at about minus 0.05 and a long tail to the right reaching 0.6 and beyond
:width: 100%
:class: pgv-figure

The dNBR histogram. The peak is not at zero, and the right-hand tail is where the burned classes come from.
```

**The shape is a corridor, not a perimeter.** Look at the severity map and the dNBR map. The detected area is a long, narrow belt
running north to south, a few kilometres wide, with sharp-edged rectangles on both sides of it and a scatter of light-yellow
specks over the rest of the box. A fire perimeter is usually one connected, irregular area shaped by wind and fuel. Straight-edged,
field-sized blocks in a narrow belt look more like land use: orchards and crops that harvested, lost their leaves or were
flooded between the two dates also lower NIR and raise SWIR. **This is an observation about the picture, not a diagnosis.** The
report has no reference perimeter and cannot tell the two apart.

```{figure} ../_static/img/case-studies/wildfire-dnbr.webp
:alt: Map of continuous dNBR values. The green background is slightly negative; a narrow north-south belt of pale orange rectangles marks the highest values
:width: 100%
:class: pgv-figure

The continuous dNBR layer. The belt and its rectangular blocks are visible without any class thresholds.
```

**Each date is one scene.** Both dates are a single Sentinel-2 acquisition, so any shadow, thin cloud or haze in either scene is
in the answer. The report's limitations say "smoke, cloud and burn scars from earlier fires can bias the result".

**Thresholds are generic.** They are simplified USGS/MTBS values for a composite burn index. They were not calibrated to this
vegetation, so "moderate" here is a dNBR range, not a measured fire effect on the ground.

### How to test it

| Check | How |
|---|---|
| Compare with an official perimeter | Overlay `deliverables/vectors/wildfire_severity.gpkg` on the agency's fire perimeter for the period. Burned area outside the perimeter is not that fire. |
| Look at the ground | Open the post-date scene or any high-resolution imagery over a "high" rectangle: burned ground is black or grey, a harvested orchard is not. |
| Change the season, not the place | Run again with a before and an after date a year apart in the **same** month. A seasonal artefact should shrink; a scar should stay. |
| Mask first | If you only care about one land cover, mask `severity_class.tif` to it (see the [land-cover page](london-landcover-habitat.md) for a reference map). |

## Change something

| To | Do |
|---|---|
| Use dates closer to the event, in the same season | `--date-before` and `--date-after` take months (`YYYY-MM`). Pick the month before the fire and the first clear month after. |
| Try other thresholds | The pipeline's thresholds are fixed at 0.10, 0.27 and 0.66. Reclassify the `dnbr.tif` it wrote with the function below. |
| Restrict to your polygon | Clip the class raster or the GeoPackage layer to your boundary after the run. |
| See what each class means | Read the [`wildfire_severity` page](../pipelines/catalog/wildfire_severity.md), which lists the classes and statistics. |

`classify_dnbr` assigns classes exactly as the pipeline does (a value on a threshold takes the higher class, a missing value is 255),
and a test compares it with the real pipeline's own raster. Pass your own thresholds:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: classify_dnbr
```

```python
# check: the class of values on and just below each threshold
import numpy as np

dnbr = np.array([-0.30, 0.0999, 0.10, 0.2699, 0.27, 0.6599, 0.66, 1.2, np.nan])
print(classify_dnbr(dnbr))
print(classify_dnbr(dnbr, thresholds=(0.15, 0.35, 0.70)))
```

```text
[  0   0   1   1   2   2   3   3 255]
[  0   0   0   1   1   2   2   3 255]
```

## Things to be careful about

- **A burned-area figure needs a method.** Say whether you summed pixels or area; they differ by 0.04 % here.
- **Cloud is not unburned.** 4.1 % of the box has no data and is excluded from every percentage, which is correct, but check that the
  empty part is not the part you care about.
- **dNBR does not know why.** Harvest, flooding, drought, phenology and fire look alike. The map is a candidate list.
- **No accuracy figure.** Nothing here was compared with reference data.

## The files it wrote

| File | What it holds |
|---|---|
| `severity_class.tif` | Classes 0 to 3, 255 for no data |
| `dnbr.tif` | The continuous dNBR |
| `deliverables/wildfire_severity_report.pdf`, `.xlsx`, `.docx` | The report |
| `deliverables/maps/*.png` | The severity map, the dNBR map, the chart and the histogram |
| `deliverables/vectors/wildfire_severity.gpkg` | Layer `burn_severity`: 222,810 polygons, 49,589.8 ha, and the `aoi` outline |
| `deliverables/vectors/geojson/*.geojson` | The same in WGS 84 |
| `run_summary.json` | The machine-readable record |

222,810 polygons for 49,589.8 ha is an average of 0.22 ha each. That many tiny polygons is what a speckled raster produces and is
worth a look before you open the layer in a desktop GIS.

## Related

- [Land cover and habitat in central London](london-landcover-habitat.md): a reference map you can use to mask a result like this.
- [`wildfire_severity`](../pipelines/catalog/wildfire_severity.md) and [`active_fire`](../pipelines/catalog/active_fire.md) (burning
  now, not the scar).
- [Reading results](../pipelines/results.md) and [troubleshooting](../getting-started/troubleshooting.md).
- [Choosing scenes](../data/scene-selection.md).
