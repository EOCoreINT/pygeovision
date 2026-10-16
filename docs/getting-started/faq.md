# Frequently asked questions

## About the tool

**What is PyGeoVision, in one sentence?**
A Python package and command-line tool that finds satellite imagery, prepares it correctly, and turns it into finished
geospatial products — flood extent, burn severity, land cover, terrain, building footprints, and more — with maps, tables
and reports, plus the machinery to train, run and explain neural networks on imagery.

**How is it different from pygeofetch?**
[pygeofetch](https://pygeofetch.readthedocs.io/) finds and downloads data from 22 providers. PyGeoVision builds on it:
it chooses scenes, applies the radiometric corrections, runs the analysis and writes the deliverables. You can use
either alone. PyGeoVision's [Data guide](../data/index.md) shows where the boundary lies.

**Do I need a GPU, or even PyTorch?**
No for the 34 *ready* pipelines — indices, terrain, hydrology, radar thresholds and published reference maps run on a
CPU and import no deep-learning library. Yes (practically) for training, and for the seven pipelines that run your
trained network.

**Does it ship trained models?**
It ships **architectures** and fetches pretrained weights where a real public checkpoint exists (see the
[model catalogue](../models/catalog.md)). It does not ship weights for tasks like "crop type" or "solar panels",
because they would be untrained and produce random output. Pipelines that need such weights refuse to run without your
checkpoint. This is deliberate.

**Which satellites and sensors does it use?**
Sentinel-2 and Landsat for optical, Sentinel-1 for radar, MODIS for fire and evapotranspiration, JAXA PALSAR for forest
change, the Copernicus DEM for terrain, plus reference products (ESA WorldCover, Dynamic World, JRC Global Surface Water,
Microsoft and Google building footprints, OpenStreetMap). The [provider reference](../data/providers.md) lists the 22
data sources; each pipeline page names what it reads.

**Is it free? Do I need accounts?**
PyGeoVision is open source. Searches use three open providers by default (`planetary_computer`, `aws_earth`,
`element84`) and most pipelines need no account. Some providers (Copernicus Data Space, USGS, NASA Earthdata, commercial
vendors) need credentials: [Authentication](../data/authentication.md).

## Using it

**Which pipeline answers my question?**
[Choosing a pipeline](../pipelines/choosing.md) maps questions to pipelines and says what each is *not* for.

**What are the dates and the bounding box?**
`--bbox min_lon min_lat max_lon max_lat` (longitude first); dates as `YYYY-MM`, `YYYY-MM-DD` or a range
(`2024-06-10/2024-06-29`). [Core concepts](concepts.md).

**Why did it use a month when I asked for a day?**
It did not: a day is searched as that day. If nothing was captured that day the result is "No usable imagery". Use a range
for events and a month for seasonal quantities.

**How big an area can I run?**
Boxes larger than one scene are fine: scenes are chosen to cover the box and mosaicked. Memory and time grow with
pixels, so start small. [Troubleshooting](troubleshooting.md#performance-and-memory).

**Can I use my own imagery instead of downloading?**
Yes. Pipelines download, but every analysis tool works on files you have: [indices](../data/indices.md),
[preprocessing](../data/preprocessing.md), [postprocessing](../data/postprocessing.md), [validation](../data/validation.md),
[inference](../inference/index.md), [OBIA](../obia/index.md).

**Can I use it without internet?**
For files you already have, yes. Searching, downloading and the reference products need network access. The unit tests
and the offline smoke test (`python tools/offline_pipeline_smoke.py`) use synthetic data and need none.

**Can I change a threshold such as the NDWI cut-off?**
Yes. Every threshold is a documented parameter: `--param ndwi_threshold=0.2` (CLI) or `ndwi_threshold=0.2`
(Python). The pipeline page lists the parameters with their defaults and the source for each default.

**Where do my results go and what do the files contain?**
In the `--output` folder: the analysis rasters, `run_summary.json`, and a `deliverables/` folder with maps, vectors and
reports. See [Deliverables](../reporting/index.md).

**Can I get the report in another format or with my class names?**
Yes: `pygeovision report <output-folder> --formats pdf,docx --class-names …` rebuilds everything from the saved
rasters without downloading anything.

## Interpreting results

**The area is different from an official statistic. Which is right?**
Possibly both. Differences come from sensor, date, resolution, threshold and definition (for example whether "flooded"
includes permanent water). Each pipeline page lists the usual reasons and what the product cannot tell you.

**What does 255 mean in a class raster?**
No data: cloud, shadow, outside the image, or no valid observation. It is never a class and is excluded from every
percentage ([Core concepts](concepts.md#nodata-masks-and-what-no-data-means)).

**Is the result validated?**
The pipelines apply published methods with documented thresholds. They do not validate the result *for your site*. For
accuracy numbers, compare with reference data using [`postprocess accuracy`](../data/postprocessing.md) or the
[benchmark tools](../experiments/benchmark.md). Treat unvalidated maps as screening products.

**Why does the flood map disagree with the news?**
Flood maps from satellites show water *at the time of the pass*. Floods peak between passes, cloud hides the optical
view, and radar can miss flooded buildings and vegetation. See [Flood and water mapping](../examples/water.md)
and the multi-sensor pipeline page.

## Models and training

**Which model should I train for X?**
[Choosing a model](../models/choosing.md) maps tasks to architectures and says what data each needs.

**Can I fine-tune a foundation model such as Prithvi?**
Yes; see [Foundation models](../foundation-models/index.md) and [Training](../training/index.md).

**Why does a `needs weights` pipeline fail before downloading anything?**
So you do not wait for a download and then get a meaningless result. The check happens first.

## Project

**Which Python versions are supported?**
3.10 and newer (tested on 3.13).

**How do I report a bug or contribute?**
[Contributing](../project/contributing.md). Include `pygeovision doctor` and the `-v` log.

**Is there a changelog?**
[Changelog](../project/changelog.md) and the [roadmap](../project/roadmap.md).
