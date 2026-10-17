# Core concepts

Read this page once. Every other page assumes the vocabulary and the conventions below, and most "why is my
result strange?" questions are answered here.

## The three layers

PyGeoVision is built in three layers. Knowing which layer a feature lives in tells you which page documents it and
which dependencies it needs.

```{figure} ../_static/img/stack-layers.svg
:alt: PyGeoVision in three layers: data at the bottom, AI in the middle, tasks on top
:width: 100%
:class: pgv-figure

Each layer uses only the one below it.
```

- The **data layer** ([Data guide](../data/index.md)) finds scenes, downloads the bands you need, applies the
  radiometric scale factors, masks cloud and writes analysis-ready GeoTIFFs.
- The **AI layer** is optional. It needs PyTorch (`pip install "pygeovision[train]"`). Nothing in the data layer or
  in the 34 *ready* pipelines imports it.
- The **task layer** ties both together into *named, finished products*: a pipeline downloads what it needs,
  computes a defined quantity and writes rasters, vectors, a map and a report.

## Pipelines: the unit of work

A **pipeline** answers one question for one area and one time. It has a name (`flood_mapping`,
`vegetation_anomaly`, `terrain_analysis`…), a fixed set of inputs and a documented output. There are 41:

| Status | Count | Meaning |
|---|---|---|
| **ready** | 34 | Runs out of the box. The result is computed from real imagery, a DEM, radar, or a published reference map. No model to train |
| **needs weights** | 7 | Runs *your* trained network. You must pass `--checkpoint` / `checkpoint=`. PyGeoVision refuses to invent an untrained answer |

Two rules apply to every pipeline, and they are the reason the catalogue is smaller than it once was:

1. **The output is what its name says.** `flood_mapping` maps flood water; it is not a generic "change" map renamed.
   A product that cannot be computed honestly from available inputs is not offered.
2. **No hidden constants and no untrained networks.** Every threshold is a documented parameter with a literature
   source; every `needs weights` pipeline stops with a clear error if no checkpoint is given.

See the [pipeline catalogue](../pipelines/index.md) for all 41, and [Choosing a pipeline](../pipelines/choosing.md)
if you are not sure which fits.

## Study area: bounding boxes

An area of interest (AOI) is a **bounding box** in longitude/latitude (EPSG:4326) in this order:

```text
min_lon  min_lat  max_lon  max_lat        # west, south, east, north
```

```bash
--bbox -0.15 51.47 -0.10 51.52           # CLI: four numbers, no commas
```

```python
bbox = (-0.15, 51.47, -0.10, 51.52)      # Python: a 4-tuple
```

```{warning}
The order is **longitude first**. Google Maps shows *latitude, longitude*; swapping the pairs places your box in the
wrong part of the world and the search finds nothing (or finds scenes of somewhere else). Before a long run, check
the box on a map: `min_lon < max_lon`, `min_lat < max_lat`, longitudes within ±180 and latitudes within ±90.
```

Boxes larger than one satellite scene are normal: PyGeoVision selects the **fewest, clearest scenes that together
cover the box** (see [Scene selection](../data/scene-selection.md)) and mosaics them. If you need the true shape of
an administrative area rather than its box, use `client.boundary("Accra, Ghana")`
([Study-area boundaries](../data/boundaries.md)).

## Dates

Three date forms are accepted wherever a pipeline takes a date:

| Form | Example | Window searched |
|---|---|---|
| Month | `2024-06` | The whole calendar month |
| Day | `2024-06-18` | That single day |
| Range | `2024-06-10/2024-06-29` | From the first day to the last day, inclusive |

Use a **range** for events that last days (floods, fires, storms): a whole month would mix the event with the weeks
around it. Use a **month** for seasonal quantities (greenness, snow cover, temperature) where you want the best
cloud-free image the month can offer.

Bi-temporal pipelines take `date_before` and `date_after`; time-series pipelines take `dates=[...]`;
`vegetation_anomaly` takes one `date` and compares it with the same month in earlier years.

```{tip}
Cloudy tropics? Widen the window, not the cloud limit. A 2–3 week range usually contains at least one clear
Sentinel-2 pass; raising `cloud_cover_max` to 90 % lets haze and shadow into the result.
```

## Grids, pixels and units

- Rasters are written in the **coordinate system of the pipeline's working grid**, and the CRS is recorded in every
  file header. Imagery pipelines work in the projection the downloaded scenes were reprojected to (usually
  geographic degrees, EPSG:4326, or the scene's UTM zone); terrain and hydrology pipelines resample to a **metric**
  grid so that slope, distance and flow accumulation are in metres. The pipeline page says which.
- Whatever the CRS, **areas are true areas**. For a projected (metric) raster the cell area is `res_x × res_y`;
  for a geographic raster each row is measured on the WGS84 ellipsoid, because a degree of longitude shrinks with
  latitude. The report footer says "Areas measured from the output raster".
- Pixel size follows the sensor (10 m for Sentinel-2 visible/NIR, 20 m for red-edge and SWIR, 30 m for Landsat,
  10 m for Sentinel-1 GRD) unless a pipeline offers `resolution_m`, which resamples to a coarser grid for speed and
  memory.
- Every result carries `aoi_coverage_fraction`: the share of your box actually covered by the imagery used. A scene
  that covers 23 % of the box produces a result that says so; do not quote totals for the whole box when it is low.

## Nodata, masks and what "no data" means

This is the single most important convention for reading results correctly.

| Value | Where | Meaning |
|---|---|---|
| `255` | class rasters (`uint8`) | **No data**: cloud, shadow, outside the swath, or no valid observation. Not "dry", not "no change" |
| `NaN` | continuous rasters (`float32`) | No data, for the same reasons |
| `0` | class rasters | Usually the *background* class ("not water", "no change", "unburned"). Each pipeline page lists its classes |

Cloud-masked pixels are never counted as "normal". Percentages are computed over **valid** pixels and the valid share
is reported next to every figure (`valid_fraction`). If `valid_fraction` is low, the number is based on little ground:
treat it with caution or widen the dates.

## Results: `PipelineResult`

Every pipeline, from Python or the CLI, produces the same structure:

| Field | Type | Meaning |
|---|---|---|
| `pipeline` | `str` | Pipeline name |
| `success` | `bool` | `True` only if a real result was produced |
| `output_path` | `str` | The main raster (or directory for multi-date pipelines) |
| `stats` | `dict` | The headline numbers — areas, fractions, thresholds actually used |
| `metadata` | `dict` | Inputs: bbox, dates, scenes used, parameters |
| `error` | `str \| None` | Why it failed, when `success` is `False` |
| `duration_seconds` | `float` | Wall time |
| `deliverables` | `dict` | Where the map, vectors and reports were written |

A pipeline that cannot find usable imagery returns `success=False` with an explanatory `error`. It does not return an
empty file or a zero-filled raster.

```python
result = client.pipeline("water_bodies", bbox=bbox, date="2024-06")
if not result.success:
    raise SystemExit(result.error)
print(result.stats["water_coverage"])
```

See [Reading results](../pipelines/results.md) for how to interpret `stats` and the class rasters.

## Deliverables

Unless you pass `--no-report`, a run also writes a **deliverables** folder next to the rasters:

```text
<output>/
├── flood_extent.tif              # the analysis raster(s)
├── run_summary.json              # every figure, parameter and layer, machine-readable
└── deliverables/
    ├── flood_mapping_report.pdf  # question, method, map, tables, limitations, references
    ├── flood_mapping_report.xlsx # the same tables as sheets
    ├── flood_mapping_report.docx # editable Word report
    ├── maps/                     # PNG map(s) and charts
    └── vectors/                  # GeoPackage and GeoJSON (optionally Shapefile, KML)
```

`pygeovision report <output>` rebuilds all of this from the saved rasters without downloading anything — use it to add
a format or to name the classes of your own network. Details: [Deliverables & reports](../reporting/index.md).

## Providers and credentials

Imagery is served by **providers**. PyGeoVision ships a bridge to 22 through
[pygeofetch](https://pygeofetch.readthedocs.io/). Three need no account and are searched by default:
`planetary_computer`, `aws_earth` and `element84`. Others (Copernicus Data Space, USGS, NASA Earthdata,
commercial vendors…) need credentials: `pygeovision data auth add <provider>`. The
[provider reference](../data/providers.md) lists each one with its collections and whether it needs a key.

## Two kinds of "model"

You will meet the word *model* in two senses:

- **Pipelines** that compute an index or product (NDWI, dNBR, HAND, a reference map) contain no trained model.
- **Networks** (U-Net, YOLO, ChangeFormer, Prithvi…) are trained models from the [model registry](../models/index.md).
  The `needs weights` pipelines, `pygeovision infer`, and the labelling tools use them.

PyGeoVision ships **architectures**, and downloads pretrained weights where a real public checkpoint exists. It
does not ship weights trained on your task — you [train](../training/index.md) or
[bring](../inference/index.md) those.

## Where to go next

```{list-table}
:widths: 30 70
* - Run something now
  - [Quickstart](quickstart.md)
* - Change defaults (device, providers, cache)
  - [Configuration](configuration.md)
* - Something failed
  - [Troubleshooting](troubleshooting.md)
* - Not sure which pipeline
  - [Choosing a pipeline](../pipelines/choosing.md)
* - Decide your area and dates
  - [Choose your area and dates](../guides/your-area.md)
* - Script it, or run many areas
  - [Use it from Python](../guides/python.md), [Run many areas](../guides/automate.md)
```
