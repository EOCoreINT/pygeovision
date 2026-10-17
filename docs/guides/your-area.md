# Choose your area and dates

Every analysis starts with two decisions: **where** (a box on the map) and **when** (a month, a day or a range). Most disappointing
results trace back to one of them: a box with the coordinates swapped, a month with no clear pass, a date window that mixes an event
with the weeks around it. This guide shows how to make both decisions well and how to test them before a long run.

## Where: get a bounding box

PyGeoVision takes the area as a **bounding box** in longitude and latitude (WGS 84), always in this order:

```text
west  south  east  north
```

These are also called `min_lon min_lat max_lon max_lat`.

Longitude comes first. Web maps usually show *latitude, longitude*, so copying a pair of numbers straight from one puts your box in
the wrong part of the world.

Pick whichever of the four routes below matches what you have.

::::{tab-set}

:::{tab-item} I know the coordinates
Type them in, west first:

```bash
pygeovision channel water_bodies --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --output out/london
```

```python
bbox = (-0.15, 51.47, -0.10, 51.52)      # a tuple of four numbers
```
:::

:::{tab-item} I have a file
Any vector file your GIS can write (GeoJSON, Shapefile, GeoPackage) gives you a box in one line:

```python
import geopandas as gpd

area = gpd.read_file("my_study_area.geojson").to_crs(4326)     # make sure it is longitude/latitude
bbox = tuple(area.total_bounds)                                # (min_lon, min_lat, max_lon, max_lat)
print(bbox)
```

`to_crs(4326)` matters: a file in a projected system (UTM, British National Grid) has bounds in metres, not degrees.
:::

:::{tab-item} I have a place name
`client.boundary()` looks the name up in OpenStreetMap and returns the real outline together with its box:

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
aoi = client.boundary("Accra Metropolitan District, Greater Accra Region, Ghana")
print(aoi.bbox, round(aoi.area_km2), "km²")
```

Check that the area is what you expect before you use it: a name that matches the wrong place gives a confident wrong box
([Study-area boundaries](../data/boundaries.md) explains how to have this checked for you).
:::

:::{tab-item} I have a map
Read the extent from the map tool in WGS 84 degrees. QGIS shows a layer's extent under *Layer Properties ▸ Information*; make sure
the values are in degrees and not in metres, and copy them into the order above.
:::

::::

### Check the box before you run

Two quick checks catch most mistakes. The first computes the true area of the box; the second counts the pixels it will produce.

```python
from pyproj import Geod

west, south, east, north = bbox
assert west < east and south < north, "the box is the wrong way round (west, south, east, north)"

lons, lats = [west, east, east, west], [south, south, north, north]
area_km2 = abs(Geod(ellps="WGS84").polygon_area_perimeter(lons, lats)[0]) / 1e6
pixels_10m = area_km2 * 1e6 / (10 * 10)
print(f"{area_km2:,.0f} km², about {pixels_10m / 1e6:,.1f} million pixels at 10 m")
```

If the area is not the size you expected, the order of the coordinates is the first thing to check.

### How large can the box be?

A box can be larger than one satellite scene: PyGeoVision picks the fewest, clearest scenes that together cover it and joins them
([Scene selection](../data/scene-selection.md)). What limits the size is memory, and three limits are worth knowing:

| Limit | Value | What happens above it |
|---|---|---|
| Output grid | 250 million pixels, about 25,000 km² at 10 m | The run stops before downloading and says to use a smaller box or a coarser resolution. |
| Terrain and flow routing (flood height-above-drainage, `drainage_analysis`) | 6 million cells | The terrain step is skipped or refused. The [Nepal case study](../case-studies/flood-nepal.md) shows the message and the one argument that fixes it (`--param resolution_m=30`). |
| Scenes per run | The set that covers the box, at most four, plus up to four more to close gaps | A box that needs more is covered only in part. Read `aoi_coverage_fraction` in the result. |

The practical rule: **start small.** Run a box of 10 to 20 km on a side first. When that looks right, enlarge it, or split a very
large region into tiles and run them one after the other ([Run many areas](automate.md)).

## When: choose the dates

### Three ways to write a date

| You write | Example | PyGeoVision searches |
|---|---|---|
| A month | `2024-06` | The whole month, and uses the clearest scene or scenes |
| A day | `2024-06-18` | That day only |
| A range | `2024-06-10/2024-06-29` | From the first day to the last, inclusive |

### Match the window to the question

| Your question | Use | Why |
|---|---|---|
| What does the land look like now? | One month | The month gives the clearest scene a chance to appear. |
| What did an event (flood, fire, storm) do? | A **range** of days around the event | A whole month mixes the event with the weeks before and after. |
| What changed between two seasons or years? | Two dates, `--date-before` and `--date-after` | The first is the **baseline**, the second the later state. |
| Is this month unusual? | One date, for pipelines such as `vegetation_anomaly` | It compares the month with the same month in earlier years. |
| What is the trend over a year? | A list of dates, `--dates 2024-01,2024-04,2024-07` | One scene per date, on one grid. |

### Compare like with like

For a change between two dates, choose dates from the **same season** unless the seasons are what you want to compare. Vegetation,
water and snow all change with the calendar, and a pair of dates months apart mixes a real change with the season. The
[vegetation case study](../case-studies/london-vegetation.md) shows what a May-to-August pair does to a vegetation-loss map.

```bash
# the same month in two different years: the seasonal part cancels
pygeovision channel crop_health --bbox -0.15 51.47 -0.10 51.52 \
    --date-before 2023-07 --date-after 2024-07 --output out/crop_year_on_year
```

### How often does a satellite pass?

| Sensor | Pass every | Sees through cloud? | Used by |
|---|---|---|---|
| Sentinel-2 (optical, 10 m) | about 5 days | No | Most pipelines |
| Landsat 8 and 9 (optical, 30 m) | about 8 days together | No | Pipelines with thermal bands, long histories |
| Sentinel-1 (radar, 10 m) | 6 to 12 days | **Yes** | The radar flood pipelines |

A single day can therefore have no scene over your box, and a single clear day is rare in cloudy seasons. Give a **range of one to
three weeks** for events, and for radar always give a range, not a day.

```{admonition} Cloudy place? Widen the window, not the cloud limit
:class: tip
A range of two or three weeks usually contains one clear pass. Raising the cloud limit to 90 % instead lets haze and shadow into
the result, which changes the answer without telling you.
```

## Test your choice before you run

A search costs seconds and downloads nothing. It tells you whether the dates have usable scenes over your box.

```bash
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --collections sentinel-2-l2a --cloud-max 20
```

The table lists one row per scene with its date and cloud percentage. Three or more rows with low cloud: go ahead. Nothing or only
cloudy rows: widen the dates or choose another month. The list shows **scene-wide** cloud cover, so a scene at 5 % can still be
cloudy over your box; the run measures the cloud over your box itself and reports it as `valid_fraction`.

## After the run: did it use what you asked for?

Two numbers in every result tell you whether the choices worked:

```python
result = client.pipeline("water_bodies", bbox=bbox, date="2024-06", output_dir="out/london")
print(result.stats.get("aoi_coverage_fraction"))   # share of your box that has imagery, 0 to 1 (None if not reported)
print(result.stats.get("valid_fraction"))          # share of that imagery free of cloud and shadow
```

- `aoi_coverage_fraction` below 1: part of your box was never seen. Quote numbers for the covered part only, or choose a window
  with a pass over the rest.
- `valid_fraction` below about 0.7: the percentages describe only the visible part. Widen the dates.

[Reading results](../pipelines/results.md) lists the other checks to make before you quote a number.

## Related

- [Core concepts](../getting-started/concepts.md): bounding boxes, dates and no-data in one page.
- [Scene selection](../data/scene-selection.md): exactly how scenes are chosen and joined.
- [Study-area boundaries](../data/boundaries.md): using an administrative outline.
- [Troubleshooting](../getting-started/troubleshooting.md#finding-imagery): when the search finds nothing.
