# Reading results

Every pipeline returns the same kind of result and writes the same kind of folder. This page explains both, so you can open any
output and know what each number and file means, and what to check before you trust it.

## The result object

```python
result = client.pipeline("water_bodies", bbox=bbox, date="2024-06", output_dir="./run")
```

`result` is a `PipelineResult`:

| Field | Meaning |
|---|---|
| `pipeline` | The pipeline name (also `result.name`). |
| `success` | `True` when the analysis produced a result. |
| `error` | **Set whenever `success` is `False`.** A specific reason, never empty. |
| `output_path` | The main raster (a `pathlib.Path`), for example `ndwi.tif`. `None` on failure. |
| `stats` | The numbers: areas in hectares, shares, thresholds actually used, scene dates. Each pipeline page lists its keys. |
| `metadata` | How it was produced: dates, bands, coordinate system, scene ids, input coverage. |
| `duration_seconds` | Time the analysis took. |
| `deliverables` | Where the map, vectors and reports went (a `Deliverables` object), or `None`. |

```python
if not result.success:
    raise SystemExit(result.error)

result.stats["water_coverage"]             # 0.155
result.deliverables.reports["pdf"]         # Path to the PDF
result.deliverables.maps                   # list of PNG maps
result.deliverables.vectors["gpkg"]        # list of GeoPackages
result.deliverables.warnings               # problems met while building them
```

`result.deliverables.as_dict()` gives the same as plain strings, convenient for JSON.

### What "failure" means

A pipeline fails (rather than returning a poor-looking result) when it cannot do its job *correctly*:

| Message starts with… | Meaning | What to do |
|---|---|---|
| "No imagery found" / "No usable imagery" | Nothing matched, or every scene was rejected (cloud, partial coverage) | Widen the dates; see [Troubleshooting](../getting-started/troubleshooting.md#finding-imagery) |
| "needs your trained weights" | A *needs weights* pipeline without `--checkpoint` | Pass your checkpoint, or use the *ready* alternative named in the message |
| "… must be a year" / "date must be …" | A date in the wrong form | Use `YYYY-MM`, `YYYY-MM-DD`, a range, or a year as the pipeline's page says |
| "no Sentinel-1 scene …" | A radar window with no acquisition; the message lists the nearest ones | Use a listed date or a longer range |
| An authentication error | A provider that needs a key | `pygeovision data auth add <provider>` ([Authentication](../data/authentication.md)) |

A *success* does not mean a *good* result: always read `valid_fraction` and the limitations on the pipeline's page.

## The output folder

```text
output/
├── <main raster>.tif         # the result: classes (uint8, 255 = no data) or a continuous index (float32, NaN = no data)
├── <other layers>.tif        # change maps, indices, frequency maps … named on the pipeline's page
├── run_summary.json          # everything about the run (see below)
└── deliverables/
    ├── <name>_report.pdf     # question, method, map, tables, limitations, references
    ├── <name>_report.docx    # the same, editable
    ├── <name>_report.xlsx    # the same tables as sheets
    ├── maps/                 # PNG maps and charts
    └── vectors/
        ├── <name>.gpkg       # every layer in one GeoPackage
        └── geojson/          # one GeoJSON per layer
```

Each pipeline page has a "What you get" table with the exact file names. Reports and vectors are described in
[Deliverables](../reporting/index.md); to change class names or formats later without downloading anything, use
[`pygeovision report`](../reporting/rebuilding.md).

### Class rasters and continuous rasters

| | Class raster | Continuous raster |
|---|---|---|
| Data type | `uint8` | `float32` |
| No data | **255** | **NaN** |
| Meaning of values | Codes listed on the pipeline page and in the report legend (for example 0 dry, 1 flooded, 2 permanent water) | Physical quantity (index, metres, degrees, dB) |
| Percentages | Over valid pixels only | Mean/min/max over valid pixels only |

Open them in QGIS directly. The **grid** depends on the pipeline: the reference-product, terrain, radar and multi-sensor pipelines write a
metric grid (the UTM zone of the area, `EPSG:326xx`), while the optical index pipelines and the ones that run your network keep the
latitude/longitude grid of the downloaded imagery (`EPSG:4326`). The pipeline page and `run_summary.json` say which. Because of this,
**do not compute areas by multiplying a pixel count by a fixed number** unless the raster is in a metric CRS. Use the areas the run already
computed, or let the tools compute true areas for either kind of grid:

```python
import json

summary = json.load(open("run/run_summary.json"))
for layer in summary["layers"]:
    for c in layer["classes"]:
        print(layer["key"], c["value"], c["name"], round(c["area_ha"], 1), f"{100 * c['share']:.1f}% of valid pixels")
```

```bash
pygeovision postprocess class-stats run/flood_extent.tif      # true area per class, also for rasters in degrees
```

To work on the pixels yourself:

```python
import numpy as np, rasterio
from pygeovision.data.radiometric import real_pixel_area_ha

with rasterio.open("run/flood_extent.tif") as src:
    classes = src.read(1)                                  # uint8, 255 = no data
    pixel_ha = real_pixel_area_ha(src.bounds, src.width, src.height, src.crs)   # true mean area of one pixel
valid = classes != 255
print((classes == 1).sum() * pixel_ha, "ha flooded;", f"{valid.mean():.0%} of the box was assessable")
```

## `run_summary.json`

The complete record of a run, written next to the rasters. Keep it with the results.

| Key | Contents |
|---|---|
| `pygeovision_version`, `generated_utc` | What produced it, and when (UTC). |
| `pipeline`, `title` | The pipeline and the report title. |
| `success`, `duration_seconds` | Outcome and run time. |
| `bbox`, `aoi_area_km2` | The requested area, and its true area in km². |
| `period` | The dates covered. |
| `parameters` | **Every argument the run used**, including the ones you did not set (defaults). This is what makes a run reproducible. |
| `metadata` | Dates, bands, CRS, scene ids and `input_coverage` (below). |
| `stats` | The statistics, as in `result.stats`. |
| `layers` | One entry per raster: key, title, file, kind (`classes` or `continuous`), and per-class `value`, `name`, `color`, `pixels`, `area_ha`, `share`, `share_aoi`, or a summary (`min`, `max`, `mean`, `median`, percentiles, `valid_fraction`) for continuous layers. |
| `deliverables` | The files of the report step, and any warnings. |

`share` is the share of **valid** pixels; `share_aoi` is the share of the whole box. They differ exactly when part of the box is no data.

### Input coverage

`metadata.input_coverage` lists, for each image used:

| Key | Meaning |
|---|---|
| `image` | Which input (for example the flood image or the reference image). |
| `aoi_coverage_fraction` | Share of **your box** that the image covered. |
| `clear_fraction` | Share of the image that was clear of cloud and shadow. |
| `scenes_used`, `scenes_failed` | Scene ids that contributed, and that could not be downloaded. |

The smallest `aoi_coverage_fraction` is repeated in `stats` as `aoi_coverage_fraction`. If it is below 1, **the numbers describe only the part of
your box that has imagery**. Do not quote totals for the whole box.

## A checklist before you use a number

1. **`success`** is `True`.
2. **`valid_fraction`** is high (more than about 0.7). If not, the percentages describe only the visible part; widen the dates.
3. **`aoi_coverage_fraction`** is close to 1.
4. **The dates in the statistics** are the ones you meant (`flood_dates`, `reference_dates`, `year_before`, …). A date window can select an
   earlier or later scene than you expected.
5. **The threshold used** is the one you intended (it is in `stats` and in `parameters`).
6. **The map** looks right over a basemap, especially hills, cities and water edges, where false alarms concentrate.
7. **The limitations** on the pipeline's page apply to your case (for example radar and urban flooding).
8. **Validation.** The pipelines apply published methods; they do not validate the result *for your site*. For accuracy figures compare
   with reference data ([Validation](../data/validation.md), [`postprocess accuracy`](../data/postprocessing.md)).

## Comparing runs

Because `run_summary.json` records the parameters, two runs can be compared safely:

```python
import json
a = json.load(open("run_2023/run_summary.json"))
b = json.load(open("run_2024/run_summary.json"))

assert a["parameters"].keys() == b["parameters"].keys()
changed = {k: (a["parameters"][k], b["parameters"][k]) for k in a["parameters"] if a["parameters"][k] != b["parameters"][k]}
print(changed)                                  # only the dates should differ
print(a["stats"]["water_coverage"], b["stats"]["water_coverage"])
```

Compare shares of valid pixels, not raw areas, whenever `valid_fraction` differs between the runs.

## Several runs and a history

`pygeovision data pipeline run` records data-pipeline runs in `~/.pygeovision/pipeline_history.jsonl`; named-pipeline runs are recorded in
their own output folder, so use one folder per run. `-o ./runs/$(date +%F)` in a script keeps them apart.
