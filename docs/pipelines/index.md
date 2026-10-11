# Pipelines

A **pipeline** takes a place and a time, finds the right imagery or reference data, prepares it correctly, computes an
answer, and delivers it as rasters, vectors, a map, tables and a report. PyGeoVision ships **40 pipelines**. Each answers
one question, for example *"which areas flooded?"*, *"how much forest was lost?"* or *"how steep is this terrain?"*.

```{list-table}
:widths: 30 70
:header-rows: 1
* - You want to…
  - Go to
* - Find the pipeline for your question
  - [Choosing a pipeline](choosing.md)
* - Read about one pipeline (why, how, outputs, what to do when it fails)
  - [The catalogue of all 40](catalog/index.md)
* - Understand what a run writes and how to read it
  - [Reading results](results.md)
* - Chain search → download → inference → export in a file
  - [YAML workflows](yaml-workflows.md)
* - Know why a pipeline you remember is gone
  - [Removed pipelines](removed.md)
```

## Two kinds of pipeline

The difference matters, so it is stated on every page.

| Kind | Count | What you need | Why |
|---|---|---|---|
| **ready** | 33 | A bounding box and dates. | Each is a documented index, a terrain product, a radar or optical method, a published reference map or a published vector dataset. The answer comes from real data and a stated method. |
| **needs weights** | 7 | A checkpoint **you trained**. | The answer comes from a neural network, and PyGeoVision ships no trained weights for these tasks: a network that has not been trained produces confident-looking random output. These pipelines refuse to run without `--checkpoint`, and they check this **before** downloading anything. |

The seven that need weights are `crop_monitoring`, `deforestation`, `disaster_assessment`, `landslide_detection`,
`road_extraction`, `solar_detection` and `urban_growth`. For most of them a *ready* sibling exists that needs no training
(`forest_change` for `deforestation`, `urban_expansion` for `urban_growth`, `road_network` for `road_extraction`, and the three flood pipelines
for `disaster_assessment`). Each page says which.

## Run one

```bash
pygeovision channel list                                   # all 40, with their kind

pygeovision channel wildfire_severity \
    --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12 \
    --output ./camp_fire
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "wildfire_severity",
    bbox=(-122.6, 39.7, -121.9, 40.2),          # min_lon, min_lat, max_lon, max_lat
    date_before="2018-10",
    date_after="2018-12",
    output_dir="./camp_fire",
)
if result.success:
    print(result.output_path, result.stats["burned_area_ha"])
else:
    print(result.error)                          # says exactly what went wrong
```

The command is `channel` (not `pipeline`): the `pygeovision pipeline` group runs [YAML workflows](yaml-workflows.md), a different thing.

## What happens during a run

1. **Check the request.** Dates, the bounding box and, for *needs weights* pipelines, the checkpoint. A bad request fails
   here, before any download.
2. **Find imagery.** Open catalogues are searched for scenes that cover the box in the date window; results are cached for an hour
   ([Search and download](../data/search-and-download.md)).
3. **Select scenes.** Scenes are chosen to cover the box with the least cloud, mosaicking several when one is not enough
   ([Scene selection](../data/scene-selection.md)).
4. **Download only the bands needed**, then apply the radiometric corrections for that sensor and mask cloud and shadow
   ([Radiometry](../data/radiometry.md)).
5. **Compute** the index, terrain product, radar change, reference map or network prediction.
6. **Write the rasters** and `run_summary.json`.
7. **Build the deliverables**: map, vectors, PDF, Word and Excel reports ([Deliverables](../reporting/index.md)). A problem in this
   step never turns a good analysis into a failure; the rasters stay and the reason is recorded.

If any of steps 1–6 cannot be done correctly, the run returns `success=False` and says why. **Nothing falls back silently to a
different method**, and no pipeline returns an empty or zero-filled raster to look like a result.

(common-options)=
## Common options

Every pipeline accepts these, in addition to its own arguments (listed on its page).

| CLI | Python | Meaning |
|---|---|---|
| `--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT` | `bbox=(…)` | The area, longitude first, in degrees (WGS 84). |
| `-o`, `--output DIR` | `output_dir="…"` | Where results go (default `./pipeline_output`). |
| `--date YYYY-MM` | `date="…"` | The month (or year, for annual products) of a single-date pipeline. |
| `--date-before`, `--date-after` | `date_before=`, `date_after=` | The two dates of a comparison. `date_before` is the **reference** (before the fire, a dry season, the baseline); `date_after` is the event or the later state. |
| `--dates a,b,c` | `dates=["a","b","c"]` | The dates of a time series. |
| `--param KEY=VALUE` | `KEY=VALUE` | Any pipeline-specific argument. Values are read as JSON when possible (`0.2`, `true`, `[5,10,15]`); `5,10,15` is a list of numbers; anything else is text. Repeat for several. A misspelt name is **reported** with a suggestion instead of being ignored. |
| `--checkpoint FILE` | `checkpoint="…"` | Your trained weights. Required by the seven *needs weights* pipelines. |
| `--model NAME` | `model="…"` | Architecture of your checkpoint, where the pipeline runs a network. |
| `--source NAME` | `source="…"` | Data source (`land_cover`: `worldcover` or `dynamic_world`). |
| `--provider NAME` | `providers=[…]` | Restrict the data providers searched (repeatable). |
| `--no-report` | `report=False` | Skip the map, vectors and reports (also `PYGEOVISION_REPORT=0`). |
| `--report-formats pdf,xlsx,docx` | `report_formats=…` | Which reports to write; `none` for no reports. |
| `--vector-formats gpkg,geojson` | `vector_formats=…` | Which vector files; also `shp`, `kml`, or `none`. |
| `--class-names "A,B,C"` | `class_names=[…]` | Names for the classes `0,1,2…` of a network you trained. |
| `--report-title "…"` | `report_title="…"` | Replace the report title. |

### Dates

- **A month** (`2024-06`) searches the whole month and uses the clearest scene(s). **A day** searches that day only, so if nothing was
  captured the answer is "No usable imagery". **A range** is `2024-06-10/2024-06-25`.
- **A bare year** (`2020`) is accepted where a pipeline works with years (annual products, `forest_change`, `landcover_change`). Where
  it is used for imagery it means the middle of the year (July; June for Landsat).
- **Radar** (Sentinel-1) passes a place every 6–12 days. Give a **range** of one to two weeks, not a day.
- Older argument names (`pre_date`/`post_date`, `baseline_year`/`analysis_year`, `start_year`/`end_year`) still work.

### Cloud is no data, not a class

Cloud- and shadow-masked pixels are written as **no data** (255 in a class raster, NaN in a continuous raster) and are excluded from
every percentage and area. A pipeline never counts a cloud as "not burned", "not water" or "no change". The
statistic `valid_fraction` tells you how much of your box could be assessed. **Read it first.**

## Using your own trained model

Seven pipelines run a network you trained; `change_detection`, `water_bodies` and `land_cover` can too (`method="model"` or a model name
as `source`).

```bash
pygeovision channel building_footprints \
    --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 \
    --checkpoint runs/buildings/best.pth \
    --class-names "Background,Building"
```

The checkpoint is loaded **strictly**: the architecture, class count and band count must match what you trained, and a mismatch is an
error, never a silent fallback. A bare state dict and a full training checkpoint (`model_state_dict`) are both accepted. See
[Training](../training/index.md) to produce one and [Choosing a model](../models/choosing.md) to pick the architecture.

## Python helpers

```python
import pygeovision

pygeovision.list_all_pipelines()                      # every name
from pygeovision.ai.pipelines import pipeline_kinds, pipeline_catalogue

pipeline_kinds()                                      # {"active_fire": "ready", "crop_monitoring": "needs_weights", …}
pipeline_catalogue()["flood_mapping"].description     # the one-line description
```

Asking for a [removed pipeline](removed.md) raises a `ValueError` that quotes the reason.

```{toctree}
:hidden:
:maxdepth: 1

choosing
catalog/index
results
yaml-workflows
removed
```
