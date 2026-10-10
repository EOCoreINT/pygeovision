<div align="center">

<!-- <img src="https://raw.githubusercontent.com/EOCoreINT/pygeovision/main/icon/pygeovision_logo.png" alt="PyGeoVision logo" width="320"> -->

# PyGeoVision

**From "I need imagery of this place" to a finished map, mask or dataset — search, preprocess, model and assess in one package.**

[![PyPI](https://img.shields.io/pypi/v/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![Python](https://img.shields.io/pypi/pyversions/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Providers](https://img.shields.io/badge/PyGeoFetch-22_providers-f59e0b?style=flat-square)](https://appiahkubis14.github.io/pygeofetch-docs/)
[![Pipelines](https://img.shields.io/badge/Pipelines-25-22c55e?style=flat-square)](#-end-to-end-pipelines)
[![Models](https://img.shields.io/badge/Registered_architectures-77-a855f7?style=flat-square)](#-models-and-training)
[![Docs](https://img.shields.io/badge/Docs-online-3b82f6?style=flat-square)](https://appiahkubis14.github.io/pygeovision-docs/)

[Features](#-features) · [Install](#-installation) · [Quick start](#-quick-start) · [CLI](#-command-line) · [Pipelines](#-end-to-end-pipelines) · [OBIA](#-object-based-image-analysis-obia) · [Status](#-project-status) · [Docs](#-documentation)


# ⚠️ Work In Progress (Pre-Alpha)
This repository is currently an active research and development (R&D) playground as part of my pre-Master's roadmap for the Copernicus Master in Digital Earth. Architecture is shifting rapidly. 


**Go from "I need imagery of this place" to a finished map, mask, or dataset — without stitching together a satellite API, a preprocessing pipeline, and a separate AI platform yourself.**

</div>

---

> **Status: actively developed.** The core workflow (search → download → preprocess → run a model →
> assess) is tested, but not everything is equally mature. [Project status](#-project-status) lists what
> is partial, optional or deliberately unimplemented, so you know before you rely on it.

## Why PyGeoVision

Applying AI to satellite imagery usually means gluing together a data-download library, a preprocessing
pipeline, a modelling framework and a pile of custom code. PyGeoVision puts those steps behind one
package, so there are fewer joins in the chain for things to quietly break.

- **One search/download interface across 22 providers** (Sentinel, Landsat, Planet, Maxar, Copernicus,
  USGS and more) through [PyGeoFetch](https://appiahkubis14.github.io/pygeofetch-docs/), with credentials
  stored once in the system keyring.
- **Models that run inside the package** — segmentation, detection, change detection, classification and
  foundation-model embeddings — with no separate AI-platform account.
- **25 pipelines** from a bounding box to a finished output — 19 run out of the box, 6 run a model you trained.
- **A training stack** with fine-tuning, geospatial losses and tiled inference for full scenes.
- **Object-based image analysis** — segmentation, object features, rule/fuzzy/supervised classification
  and object-level accuracy.
- **A CLI and a Python API** that cover the same ground.

It is aimed at geospatial researchers who want satellite data and AI in one import, ML practitioners who
would rather not hand-roll provider APIs, and teams who need scriptable, schedulable workflows.

## ✨ Features

| Area | What you get |
|---|---|
| **Data** | Search and download from 22 providers; parallel, resumable downloads with checksum verification; chained post-processing (`unzip`, `reproject`, `compress`, `resample`, `clip`, `merge`, `ndvi`/`ndwi`, `pan-sharpen`, `atmospheric:sen2cor`, `cog`); YAML pipelines with cron scheduling |
| **Preprocessing** | Cloud masking, normalisation, band stacking and AI-ready preparation (`prepare_for_ai`); 21 spectral indices and transforms (NDVI, EVI, NBR, tasselled cap, PCA, …); a data validator that can repair common problems |
| **Segmentation** | Buildings, water (NDWI), SAM, or your own model |
| **Detection** | Generic objects, ships, cars (YOLO — needs the optional `ultralytics` package) |
| **Classification** | CLIP zero-shot scene classification; ESA WorldCover land cover |
| **Change detection** | Change vector analysis (Otsu) with no weights, or ChangeFormer / BIT / DSAMNet with your own trained checkpoint |
| **Foundation models** | Prithvi-EO, DINOv2, SAM, CLIP/RemoteCLIP, DOFA, TESSERA and AlphaEarth embeddings |
| **Labeling** | Generate training labels from OpenStreetMap, Microsoft and Google building footprints, ESA WorldCover, Dynamic World, SAM and foundation models |
| **OBIA** | Scale-controlled segmentation, object features, rule/fuzzy/supervised classification, object-level accuracy |
| **Explainability** | Grad-CAM / Grad-CAM++, uncertainty estimation, attention maps, SHAP |
| **Monitoring** | Drift detection, performance tracking and alerting for deployed models |
| **Deployment** | ONNX Runtime, NVIDIA Jetson, and AWS SageMaker / Azure ML / GCP Vertex AI deployers |

## 📦 Installation

```bash
pip install pygeovision                     # core: data search/download + band-math inference
pip install "pygeovision[geo]"              # rasterio, geopandas, rioxarray, ...
pip install "pygeovision[train]"            # PyTorch, timm, segmentation-models-pytorch, albumentations, ...
pip install "pygeovision[foundation]"       # transformers, scikit-learn, umap, faiss (foundation models)
pip install "pygeovision[obia]"             # scikit-image, scikit-learn, rasterio, geopandas (OBIA)
pip install "pygeovision[report]"           # map, vector layers and PDF/Word/Excel reports for every pipeline run
pip install "pygeovision[all]"              # everything except cloud/edge
```

Other extras: `vlm` (CLIP), `labeling`, `xai`, `monitoring`, `advanced`, `geo3d`, `viz`, `edge`
(ONNX Runtime), `cloud` (AWS/Azure/GCP SDKs), `dev` (tests and linters).

**Requirements:** Python 3.10+. PyTorch is only needed for training and deep-model inference; search,
download, NDWI/NDVI segmentation and the OBIA module work without it.

## ⚡ Quick start

Search, download, prepare and classify one scene:

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

results = client.search(bbox=(38.6, 8.9, 38.95, 9.15),
                        date_range=("2024-01-01", "2024-04-30"),
                        cloud_cover_max=15)
scene = client.download(results[:1], output_dir="./data",
                        bands=["B02", "B03", "B04", "B08", "B11", "B12"])[0]

ready = client.prepare_for_ai(scene.path, model_type="foundation",
                              output_path="./data/ready.tif")
client.classification.land_cover(ready["output_path"], output_path="./land_cover.tif")
```

Segmentation, detection and change detection follow the same pattern:

```python
client.segmentation.buildings("scene.tif", output_path="buildings.tif")
client.detection.ships("port.tif", output_path="ships.tif")
client.change.detect("2020.tif", "2024.tif", output_path="change.tif")
```

`client.validator` checks rasters and can auto-fix common problems before they reach a model.

## 🖥️ Command line

Everything above is also available from the shell:

```bash
pygeovision data search --bbox -74.1 40.6 -73.7 40.9 --date 2024-06 --satellite sentinel-2
pygeovision data download --bbox -74.1 40.6 -73.7 40.9 --date 2024-06 --output ./data/

pygeovision ai segment buildings --input scene.tif --output buildings.tif
pygeovision ai change --before 2020.tif --after 2024.tif --output change.tif

pygeovision channel list
pygeovision channel building_footprints --bbox -0.15 51.47 -0.10 51.52 --date 2024-06

pygeovision status    # what is installed and working
pygeovision doctor    # diagnose a broken setup
```

`pygeovision --help` shows the full tree: `data`, `ai`, `channel`, `pipeline` (YAML workflows), `models`,
`zoo`, `label`, `explain`, `monitor`, `edge`, `cloud`, `vlm`, `timeseries`, `validate`, `preprocess`,
`indices`, `postprocess`, `benchmark`, `datasets`, `infer` and `obia`.

[`CLI_TESTING_GUIDE.md`](CLI_TESTING_GUIDE.md) has a runnable example for every command, including all 25
pipelines, and says which ones need network access, credentials or your own trained weights.
`python tools/verify_cli_guide.py` runs the offline examples on synthetic data.

## 🛰️ End-to-end pipelines

The catalogue holds **25 pipelines**, all runnable from Python (`client.pipeline(name, ...)`) and from the CLI
(`pygeovision channel <name>`). `pygeovision channel list` shows each one and its kind:

- **19 run out of the box.** Each is a documented index, terrain product or published reference map, computed
  on valid pixels only (cloud- and shadow-masked pixels are nodata, never a class).
- **6 need a model you trained** (`building_footprints`, `solar_detection`, `crop_monitoring`,
  `disaster_assessment`, `deforestation`, `urban_growth`). PyGeoVision ships no weights for them; without
  `checkpoint=` they refuse to run, before downloading any imagery.

Every run also leaves a **presentation-ready deliverables folder** (needs `pip install "pygeovision[report]"`):

```
<out>/deliverables/<pipeline>_report.pdf | .docx | .xlsx   cover, key figures, map, tables, method, limitations, references
<out>/deliverables/maps/*.png                              map (scale bar, north arrow, legend with areas) and charts
<out>/deliverables/vectors/<pipeline>.gpkg                 GeoPackage: aoi + classified polygons + per-class summary
<out>/deliverables/vectors/geojson/ | shapefile/ | kml/    the same layers in other formats
<out>/run_summary.json                                     machine-readable statistics
```

Areas in the report are measured on the output rasters on the WGS84 ellipsoid and agree with the statistics the
pipeline prints. Choose formats with `--report-formats pdf,docx,xlsx` and `--vector-formats gpkg,geojson,shapefile,kml`,
switch off with `--no-report` (or `PYGEOVISION_REPORT=0`), and rebuild from a finished run with
`pygeovision report <out>`. In Python: `client.pipeline("flood_mapping", ..., report_formats=["pdf"])`
and `result.deliverables`.

| Pipeline | Output |
|---|---|
| `water_bodies` | Water extent (NDWI), or your model with `method="model"` |
| `change_detection` | Change mask between two dates (change vector analysis, or your model) |
| `land_cover` | ESA WorldCover 2021 reference map for the area (static; not a change product) |
| `crop_health` · `vegetation_indices` · `water_quality` | NDVI anomaly, NDVI/EVI/NDWI time series, chlorophyll proxy (NDCI) |
| `flood_mapping` · `coastal_monitoring` · `wetland_mapping` | Flood extent vs permanent water, shoreline change, wetland water/vegetation classes |
| `wildfire_severity` | dNBR burn severity (USGS classes) |
| `snow_cover` · `glacier_monitoring` | NDSI snow map, glacier-area change |
| `urban_heat_island` · `land_surface_temperature` · `volcano_monitoring` | Landsat surface temperature and thresholded hot spots |
| `biodiversity_hotspot` | Land-cover diversity (Shannon) hotspots |
| `solar_potential` · `archaeological_site` · `wind_farm_siting` | DEM-based terrain suitability and relief layers (OpenTopography key needed) |
| `building_footprints` · `solar_detection` · `crop_monitoring` | Segmentation with **your** checkpoint |
| `disaster_assessment` · `deforestation` · `urban_growth` | Bi-temporal change maps with **your** siamese checkpoint |

The other 24 of the original 49 were removed before launch because they could not produce a correct result
(untrained networks, placeholder indices, invented constants). Each removal has a documented reason: see
[removed pipelines](docs/pipelines/removed.md).

```python
client.pipeline("wildfire_severity", bbox=bbox, output_dir="out", date_before="2018-10", date_after="2018-12")
client.pipeline("building_footprints", bbox=bbox, output_dir="out", date="2024-06",
                checkpoint="runs/buildings/checkpoints/best.pth")
```

## 🧠 Models and training

- **77 registered architectures** (`pygeovision models list`): SegFormer, Mask2Former, DETR/RT-DETR,
  ChangeFormer, ViT, SAM, CLIP, DINOv2, Prithvi, DOFA and more, plus a 14-model native hub
  (U-Net, DeepLabV3+, FCOS, RetinaNet, Siamese U-Net, …). 63 registry entries build real models; 14 are
  explicit placeholders that raise a clear error — see [Choosing a model](docs/models/choosing-a-model.md).
- **Fine-tuning:** start from a checkpoint, freeze the backbone, and train with geospatially aware
  augmentation; detection training reports real mAP.
- **Losses:** Dice, Focal, Tversky, Boundary-aware, Lovász, Combo and a geospatial mixed loss; GIoU/DIoU/CIoU/SIoU
  for boxes.
- **Scale:** mixed precision, gradient accumulation and distributed multi-GPU training.
- **Tiled inference:** Gaussian-blended tiles so a model trained on small chips runs on a full scene;
  ONNX export; ensembling and prediction validation.

## 🧩 Object-based image analysis (OBIA)

Analyse imagery as objects instead of pixels: segment, describe, classify, assess.

```python
from pygeovision import obia

scene = obia.make_demo_scene()                       # synthetic scene with known truth (offline)
seg   = obia.segment(scene.raster, "multires", scale=20)
feats = obia.extract_features(seg.labels, scene.raster,
                              extra_layers=obia.spectral_layers(scene.raster))
rules = obia.RuleBasedClassifier([("water", "mean_ndwi > 0.3"),
                                  ("bare_field", "mean_ndvi <= 0.55"),
                                  ("cropland", "mean_ndvi > 0.55")])
result = rules.classify(feats)
```

It includes six segmentation methods (including Baatz & Schäpe-style region merging with a scale parameter —
*not* a clone of eCognition), spectral/shape/GLCM-texture/context features, rule-based, fuzzy and supervised
classifiers, and object-level accuracy and segmentation-quality metrics. Try it with
`pygeovision obia demo ./obia_demo`, work through the tutorial scripts in [`examples/obia/`](examples/obia),
and see [docs/obia](docs/obia/index.md) for the guide and its limitations.

## 📌 Project status

Things worth knowing before you build on it:

- **Five of the 79 registry entries are placeholders** (`dinov3_segmentor`, `changestar-r18`, `satlas-pretrain`,
  `chatearthnet`, `lisat-7b`): they are discoverable but deliberately raise `NotImplementedError` instead of
  silently building a different architecture. The other 74 build real networks; pretrained loads need network
  access (and, for gated repos such as DINOv3, an accepted licence) and raise `ModelLoadError` rather than
  falling back to random weights unless you pass `allow_random_init=True`.
- **Everything in the deep-learning layer was verified offline** - synthetic rasters, randomly initialised tiny
  models and stand-ins - never with pretrained weights, a GPU or real scenes. Treat first runs on your data as
  validation runs.
- **YOLO detection needs `ultralytics`**, which is not installed by any extra: `pip install ultralytics`.
- **Pipelines were tested on synthetic rasters with known answers, not on live imagery.** Optical pipelines
  are still limited by clouds, haze and shadow; pick dates in the same season.
- **`client.detection.ships/cars` use a general-purpose (COCO) detector** that is not suited to 10 m pixels.
- **Inference commands run your own trained network.** `infer predict|batch`, `ai infer`, `ai segment|detect
  custom` and the `changeformer|bit|dsamnet` methods of `ai change` require `--checkpoint`; the registry holds
  architectures, not trained weights. For change maps from a bounding box use
  `pygeovision channel change_detection`.
- **Some features download model weights on first use** (for example ImageNet or Hugging Face weights);
  fully offline environments need them cached beforehand.
- **Removed from earlier versions:** the inference server, enterprise/RBAC layer, SAR/InSAR processing and the
  planner agent are not part of the package. See the [roadmap](docs/reference/roadmap.md) for the history and
  the direct replacements.

## 🧪 Testing

```bash
pip install "pygeovision[dev,train,geo,obia]"
pytest tests/
```

With the full stack installed, the offline suite (`tests/test_live.py` excluded) runs about 1,300 tests, all of which
pass without network access, GPU or pretrained weights. To exercise every CLI command and all 25 pipelines by hand,
follow [CLI_TESTING_GUIDE.md](CLI_TESTING_GUIDE.md).

## 📋 Documentation

Full documentation is at **https://appiahkubis14.github.io/pygeovision-docs/**. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## 🤝 Contributing

Contributions are welcome — see [docs/contributing.md](docs/contributing.md) to get started.

## 📄 License

PyGeoVision is free and open-source software, released under the [MIT License](LICENSE).

## Acknowledgements

The satellite data layer is built on **[PyGeoFetch](https://appiahkubis14.github.io/pygeofetch-docs/)**,
a universal satellite data pipeline that handles search, download, authentication, caching and pipeline
orchestration. The modelling, training, labeling and OBIA layers are implemented inside PyGeoVision itself.
