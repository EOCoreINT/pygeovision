# Use PyGeoVision from Python

Everything the command line does is also available from Python, and a few things (training your own network, working with arrays)
are easier there. This page shows how to run an analysis, read what comes back, and use the tools on files you already have. It
covers what you need to get work done; every part links to the page that explains it in full.

## The client

One object gives access to everything. Create it once.

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
```

It takes no required arguments. Settings such as the cache folder and the data providers come from your
[configuration](../getting-started/configuration.md). Heavy parts (PyTorch, model weights) load only when you first use them, so
creating the client is fast and works on a machine without a GPU.

## Run an analysis

`client.pipeline()` runs any of the [41 pipelines](../pipelines/catalog/index.md) by name. It takes the same inputs as the
`pygeovision channel` command: a box, the dates, an output folder, and any pipeline-specific argument as a keyword.

```python
result = client.pipeline(
    "water_bodies",
    bbox=(-0.15, 51.47, -0.10, 51.52),      # west, south, east, north
    date="2024-06",
    output_dir="out/london",
)
```

The call **does not raise** when the analysis cannot be done. It returns a result with `success=False` and the reason in `error`, so
a script can decide what to do. Test it every time:

```python
if not result.success:
    raise SystemExit(result.error)          # "No usable imagery ...", "needs your trained weights ...", and so on
```

```{note}
The one exception is a pipeline that needs your trained network and has no `checkpoint=`. It raises `ModelWeightsRequired` before
anything is downloaded, because there is nothing it could return that would be honest.
```

### What comes back

| Attribute | What it holds |
|---|---|
| `result.success` | `True` when a real result was produced |
| `result.error` | The reason, when `success` is `False` |
| `result.output_path` | The main raster, as a path |
| `result.stats` | The headline numbers: areas in hectares, shares, thresholds used. The keys differ by pipeline and are listed on its page |
| `result.metadata` | How the result was made: dates, scenes, coordinate system, input coverage |
| `result.deliverables` | Where the map, the vector files and the reports were written |

```python
print(result.output_path)                       # out/london/ndwi.tif, the main (continuous) raster
print(result.stats)                             # the numbers
print(result.deliverables.reports["pdf"])       # the PDF report
```

[Reading results](../pipelines/results.md) explains each number and lists what to check before you quote one.

### Open the outputs

A pipeline writes ordinary GeoTIFFs and GeoPackages, so any Python geospatial tool reads them. The main raster is
`result.output_path`; the other layers sit in the same folder, and each pipeline's page lists them. For `water_bodies` the main
raster is the continuous index (`ndwi.tif`) and the class map is `water_mask.tif`. In a class raster, **255 means no data**
(cloud, shadow, outside the imagery), not a class:

```python
from pathlib import Path

import numpy as np
import rasterio

mask_file = Path(result.output_path).parent / "water_mask.tif"
with rasterio.open(mask_file) as src:
    classes = src.read(1)                       # a 2-D array of class numbers
    print(src.crs, src.res)                     # the coordinate system and the pixel size

valid = classes != 255
for value in np.unique(classes[valid]):
    share = (classes[valid] == value).mean()
    print(f"class {value}: {share:.1%} of the valid pixels")
```

Vector layers open with GeoPandas. The list of layers and the file names are on each pipeline's page:

```python
import geopandas as gpd

layers = result.deliverables.vectors            # {"gpkg": [Path, ...], "geojson": [...]}
print(layers)
gdf = gpd.read_file(layers["gpkg"][0])         # a GeoPackage can hold several layers: name one with layer="..."
print(gdf.head())
```

## Find and download imagery yourself

You do not need to for a pipeline, which finds and downloads what it needs. Do it directly when you want the scenes themselves, for
your own analysis or to train a model.

```python
scenes = client.search(
    bbox=(-0.15, 51.47, -0.10, 51.52),
    date_range=("2024-06-01", "2024-06-30"),
    collections=["sentinel-2-l2a"],
    cloud_cover_max=20,
)
print(len(scenes), "scenes")

chosen = client.select_covering_scenes(scenes, (-0.15, 51.47, -0.10, 51.52), max_scenes=2)
files = client.download(chosen, output_dir="data/london", bands=["B02", "B03", "B04", "B08", "SCL"])
```

[Search and download](../data/search-and-download.md) describes the fields of the results, and
[Providers](../data/providers.md) lists the 22 sources and which need an account.

## Work on files you already have

The same client carries the tools for rasters you bring yourself. Each takes and returns file paths, so they chain.

```python
# 1. check that a raster is fit to use
report = client.validator.validate("scene.tif", required_bands=4, value_range=(0, 1))
print(report.summary())

# 2. compute an index (band numbers start at 1)
client.indices.ndvi("scene.tif", red_band=3, nir_band=4, output_path="ndvi.tif")

# 3. measure and vectorise a class raster
print(client.postprocess.class_statistics("out/london/water_mask.tif"))
client.postprocess.vectorise("out/london/water_mask.tif", "water.geojson", target_class=1, min_area_m2=2000)
```

`class_statistics` returns one entry per class with `pixels`, `area_ha`, `area_km2` and `pct`, counted over the valid pixels only.
`vectorise` writes one polygon per connected patch with its true area in `area_m2`, and `min_area_m2` drops patches smaller than
that.

| You want to | Use | Explained in |
|---|---|---|
| Check a raster before a model sees it | `client.validator` | [Validation](../data/validation.md) |
| Stack, clip, mask, rescale, resample, mosaic | `client.preprocess` | [Preprocessing toolbox](../data/preprocessing.md) |
| Compute NDVI, NDWI, NBR and 18 more indices | `client.indices` | [Spectral indices](../data/indices.md) |
| Polygons, areas, zonal statistics, accuracy, COG, export | `client.postprocess` | [Postprocessing toolbox](../data/postprocessing.md) |
| Make a raster model-ready in one call | `client.prepare_for_ai()` | [AI-ready data](../data/ai-ready.md) |
| Use the real outline of a district | `client.boundary("name")` | [Study-area boundaries](../data/boundaries.md) |

## Train and run your own network

Training and prediction are Python-first. The trainer, the model registry and the explainability tools are separate classes, not
methods of the client; their pages show the imports.

| You want to | Start at |
|---|---|
| Build a training set from OpenStreetMap or a reference map | [Labeling](../labeling/index.md) |
| Train a model | [Training](../training/index.md) |
| Run a trained model over a scene | [Inference](../inference/index.md) |
| Use a pretrained foundation model | [Foundation models](../foundation-models/index.md) |
| Explain a prediction | [Explainability](../explainability/index.md) |

[Train a model and use it](own-model.md) strings these together in the order you would do them.

## Check what is installed

```python
status = client.status()                # versions of PyGeoVision, pygeofetch, PyTorch, rasterio ...
print(status["pygeovision_version"], status["torch"])
```

`pygeovision doctor` on the command line does the same and also tests the data providers
([Diagnostics](../getting-started/diagnostics.md)).

## Related

- [Run many areas](automate.md): loops, tables of results and schedules.
- [Use the results](use-results.md): GIS, reports and sharing.
- [Reading results](../pipelines/results.md): what every field of a result means.
- [Exceptions](../reference/exceptions.md): the error types and when each is raised.
