# Quickstart

Ten minutes from an empty environment to a finished map, a table of figures and a report. You will install the
package, check that it works, run one of the 41 pipelines, look at what it wrote, and then run the same thing from
Python. Everything here uses open data and needs no account.

## 1. Install and check

```bash
python -m venv .venv && source .venv/bin/activate
pip install "pygeovision[geo,report]"      # rasters, vectors and the map/PDF/Word/Excel reports
pygeovision --version
pygeovision doctor
```

`doctor` reports whether each component is available. A healthy install prints `7/7 components healthy`:

```text
  ✓  pygeovision
  ✓  pygeofetch          v2.6.3.0.4
  ✓  torch               v2.14.1+cu130      # only shown if PyTorch is installed
  ✓  rasterio            v1.5.2
  ✓  geopandas           v1.2.0
  ✓  credentials         0 provider(s) with stored credentials
  ✓  cache               1 entries
```

A ✗ names the missing piece; see [Installation](installation.md) for which extra provides it. PyTorch is **not**
needed for the pipelines in this quickstart.

## 2. See what is available

```bash
pygeovision channel list
```

```text
41 PyGeoVision pipelines (34 ready, 7 need your trained weights):

  crop_health                ready         NDVI drop between two dates (vegetation stress / loss anomaly)
  flood_mapping              ready         Flooded area = MNDWI water on the flood date minus permanent water from a dry reference
  terrain_analysis           ready         Elevation, slope, aspect, hillshade and ruggedness (TRI) from the Copernicus 30 m DEM
  water_bodies               ready         Surface water mask from NDWI (green, NIR); or your own trained model
  crop_monitoring            needs_weights Crop-type segmentation with your trained model (checkpoint required)
  …
```

*ready* pipelines compute their answer from real imagery, a DEM, radar or a published map. *needs_weights* pipelines
run a network you trained and stop with a clear message if you do not pass `--checkpoint`
([Core concepts](concepts.md)). The command lists all 41.

## 3. Run your first pipeline

Surface water over part of London, June 2024. The box is `min_lon min_lat max_lon max_lat`, longitude first.

```bash
pygeovision channel water_bodies \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date 2024-06 \
    --output ./first_run
```

PyGeoVision searches the open catalogues, picks the clearest scene(s) that cover the box, downloads only the bands
needed (green and near-infrared plus the cloud layer), computes NDWI, thresholds it, and writes the results. The
first run downloads imagery and takes a minute or more depending on your connection; the search itself is cached for
an hour.

**What you see:**

```text
Running pipeline 'water_bodies'...
  bbox=(-0.15, 51.47, -0.1, 51.52) | output=./first_run

✓ pipeline complete
  Output: ./first_run/ndwi.tif
  Stats:
    water_coverage: 0.1550
    valid_fraction: 1.0000
    ndwi_threshold: 0.3000

  Deliverables: ./first_run/deliverables
    PDF report    water_bodies_report.pdf
    Excel report  water_bodies_report.xlsx
    Word report   water_bodies_report.docx
    Map           maps/water_bodies_map.png
    Vectors gpkg  water_bodies.gpkg
    Vectors geojson water_bodies_aoi.geojson, water_bodies_water_bodies.geojson
```

```{note}
The figures above come from the **synthetic scene used by the offline test harness**, to show the layout without
publishing a run over a real place. Your numbers describe your scene.
```

## 4. Read the result

```text
first_run/
├── ndwi.tif                  # the continuous index (float32, NaN = no data)
├── water_mask.tif            # the classes (uint8: 1 water, 0 not water, 255 no data)
├── run_summary.json          # every figure, parameter, scene and layer
└── deliverables/
    ├── water_bodies_report.pdf   # question, method, map, tables, limitations, references
    ├── water_bodies_report.xlsx  # the same tables as sheets
    ├── water_bodies_report.docx  # editable Word report
    ├── maps/                     # PNG maps and charts
    └── vectors/
        ├── water_bodies.gpkg     # all layers in one GeoPackage
        └── geojson/              # one GeoJSON per layer
```

Open `water_mask.tif` or `water_bodies.gpkg` in QGIS. In the statistics:

- `water_coverage` is the share of **valid** pixels classed as water (0.155 = 15.5 %).
- `valid_fraction` is the share of the box that was usable (not cloud, shadow or outside the image). **Check this
  first.** If it is low, the percentages describe only the visible part.
- `ndwi_threshold` is the cut-off actually used.

How to interpret each pipeline's numbers is on its page in the [catalogue](../pipelines/index.md); the general rules
(255 means no data, areas are true areas) are in [Core concepts](concepts.md).

## 5. The same thing in Python

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "water_bodies",
    bbox=(-0.15, 51.47, -0.10, 51.52),
    date="2024-06",
    output_dir="./first_run",
)

if not result.success:
    raise SystemExit(result.error)          # a clear reason, never an empty file

print(result.output_path)                   # …/ndwi.tif
print(result.stats["water_coverage"])       # 0.155
print(result.deliverables)                  # where the report, map and vectors went
```

`result` is a `PipelineResult` ([Reading results](../pipelines/results.md)). Pass `report=False` to skip the map and
reports when you only want the rasters.

## 6. Try the other kinds of pipeline

```bash
# Two dates: burn severity after a fire
pygeovision channel wildfire_severity --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12 --output ./camp_fire

# Terrain: no dates needed
pygeovision channel terrain_analysis --bbox -0.35 5.55 -0.10 5.75 --output ./accra_terrain

# A published reference map
pygeovision channel land_cover --bbox -0.35 5.55 -0.10 5.75 --output ./accra_lc
```

Not sure which one answers your question? [Choosing a pipeline](../pipelines/choosing.md) maps questions to pipelines.

## 7. Work on files you already have

You do not need the download machinery to use the analysis tools. For any raster on disk:

```bash
pygeovision validate run scene.tif                         # is it fit for a model?
pygeovision indices compute scene.tif ndvi --output ndvi.tif --red 3 --nir 4
pygeovision postprocess class-stats classes.tif            # area of every class
pygeovision postprocess vectorise classes.tif --output polygons.geojson --target-class 1 --min-area 100
```

See the [Data guide](../data/index.md) for what each does and the pitfalls (scaling, nodata, units).

## 8. Re-run only the report

Changed your mind about class names or want another format? Rebuild without downloading anything:

```bash
pygeovision report ./first_run
```

## Where next

```{list-table}
:widths: 32 68
* - Understand the conventions
  - [Core concepts](concepts.md)
* - Change providers, cache, logging
  - [Configuration](configuration.md)
* - A run failed or looks wrong
  - [Troubleshooting](troubleshooting.md)
* - Find the right pipeline
  - [Choosing a pipeline](../pipelines/choosing.md)
* - Use your own data and models
  - [Data guide](../data/index.md), [Models](../models/index.md), [Training](../training/index.md)
* - See complete workflows
  - [Examples](../examples/index.md)
```
