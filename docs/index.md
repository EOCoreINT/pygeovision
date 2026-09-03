# PyGeoVision

[![PyPI](https://img.shields.io/pypi/v/pygeovision.svg)](https://pypi.org/project/pygeovision/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/pygeovision/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-1150_passing-brightgreen.svg)](architecture.md)
[![Docs](https://img.shields.io/badge/docs-honest_verification_status-teal.svg)](architecture.md)

**A geospatial AI platform combining PyGeoFetch's satellite data access with native, self-contained AI inference — no external AI platform, account, or API key required for model inference.**

```bash
pip install pygeovision
```

---

## 📖 Introduction

PyGeoVision is a Python package that combines two things normally kept
in separate projects: **satellite data access** (search, download,
provider authentication, radiometric/atmospheric preprocessing — via
[PyGeoFetch](https://pypi.org/project/pygeofetch/), a real, independent
PyPI dependency) and **AI inference** (model architectures, training
utilities, and 51 task-specific pipelines, implemented natively in
pure PyTorch, HuggingFace Transformers, and timm — with no dependency
on any separate geospatial-AI platform).

The package is organized around a real client object
(`pygeovision.PyGeoVision`) that exposes both layers through one
consistent interface, plus a CLI (`pygeovision`) for running any of the
package's pipelines, models, and workflows without writing Python.

This documentation has been through a real, systematic verification
audit — every claim you read here is labeled as either independently
tested this cycle (with real or hand-checkable synthetic data) or not
yet independently verified, rather than presented as uniformly
authoritative. See [Verification status](#verification-status-read-this-first)
below, and [Architecture](architecture.md) for the full picture.

## 📝 Why PyGeoVision

Applying AI to satellite imagery normally means gluing together a
provider-specific data client, a separate preprocessing library, a
model zoo, and a training/inference framework — each with its own
conventions, and each requiring you to hand-write the radiometric
scaling, band alignment, and cloud-masking logic that sits between
"downloaded scene" and "model-ready tensor." PyGeoVision's specific
niche is owning that middle layer directly: real, tested radiometric
scaling (Landsat Collection 2 and Sentinel-2 L2A, mission-aware band
aliases), real cloud masking, real bi-temporal grid alignment for
change-detection tasks, and a model registry that can be called by a
single name (`get_model("segformer-b2", ...)`) rather than an
architecture-specific import.

It is not a general-purpose geospatial-AI framework competing on
architecture count — see [Verification status](#verification-status-read-this-first)
for an honest accounting of how much of the 121-model registry and
51-pipeline catalog is independently confirmed working versus
not-yet-verified or genuinely unimplemented. If you need a
comprehensive, actively-maintained geospatial-AI toolkit with a large
example library today, [GeoAI](https://opengeoai.org) is a mature,
well-documented alternative worth comparing against directly.

## 🚀 Key Features

### 🛰️ Satellite Data Access

- Unified search and download across ~24 provider integrations (Sentinel, Landsat, Planet, Maxar, Copernicus, USGS, NASA, JAXA, and more) via [PyGeoFetch](https://pypi.org/project/pygeofetch/)
- Real, tested radiometric scaling — Landsat Collection 2 (`SR_SCALE=0.0000275, offset=-0.2`) and Sentinel-2 L2A (`/10000`), mission-aware band aliases (Landsat 7/5 ETM+/TM vs. Landsat 8/9 OLI genuinely differ in band-to-wavelength mapping)
- Real cloud masking (QA_PIXEL / SCL), real bbox cropping, real geodesic area calculation (`pyproj`-based — fixes a confirmed bug where naive pixel-resolution math was wrong by ~10 billion times on `EPSG:4326` output)
- Credential handling via system keyring; parallel downloads with checksum verification and resume support

### 🧠 AI Task Pipelines — all 51 individually verified

- 10 original pipelines (building footprints, land cover, change detection, crop monitoring, deforestation, disaster assessment, solar detection, urban growth, water bodies, carbon estimation) — real radiometric handling, bi-temporal grid alignment
- 13 pipelines converted from stubs to real implementations this audit cycle — real dNBR (wildfire severity), NDSI (glacier/snow), MNDWI+EVI (wetlands), Landsat thermal LST (urban heat/volcano monitoring), NDCI (water quality), Shannon diversity index (biodiversity), and more
- 12 pipelines that honestly raise a specific `NotImplementedError` rather than a fake success — each with an individually-verified reason (no real data source exists, or the required algorithm is genuinely too specialized to trust yet)
- See [Task Pipelines](api/task_pipelines.md) for the complete, per-pipeline reference

### 🗂️ Model Registry — 121 architectures, systematically tested

- Classification, detection, segmentation, change detection, foundation models (Prithvi-EO-2.0, DINOv3), VLMs (CLIP, Moondream), 3D/point cloud, time series, super-resolution
- Every model individually build-tested and forward-pass tested this cycle, not just import-checked — see [Model Layer](api/models.md) for the real, verified breakdown (working / fixed / network-blocked / needs-verification / genuinely unimplemented)
- A real device-placement bug (`.to()`) affecting 9 models, found across two independent code paths, fixed and regression-tested

### 🏷️ Auto-Labeling & Detection

- ESA WorldCover, OSM, Microsoft/Google Buildings, Dynamic World, SAM-based auto-labeling
- Real object detection (`client.detection.ships/cars`) — a severe COCO-class mislabeling bug found and fixed this cycle

### 🖥️ CLI & Serving

- `pygeovision channel <name>` — run any of the 51 verified pipelines directly from the command line
- 23 total CLI entry points (20 command groups + `channel`/`status`/`doctor`) covering data search/download, model management, inference, labeling, explainability, monitoring, edge/cloud deployment, VLMs, time series, benchmarking, and raster preprocessing
- FastAPI-based model serving, ONNX export for edge deployment

## Verification status — read this first

This documentation distinguishes between code independently tested
against real or hand-checkable synthetic data this audit cycle, and
code that exists but hasn't been independently verified yet. See
[Architecture](architecture.md) for the complete picture, including
every real bug found and fixed, and exactly what remains unaudited.

## 📦 Installation

### Using pip

```bash
pip install pygeovision
```

### With extras

PyGeoVision's dependencies are split into optional extras so you only install what you need:

```bash
pip install "pygeovision[geo,train]"    # recommended minimal install: raster/vector stack + PyTorch training
pip install "pygeovision[all]"          # geo, viz, train, serve, labeling, vlm, xai, foundation, advanced, monitoring, geo3d, enterprise
```

| Extra | What it adds |
|---|---|
| `geo` | `rasterio`, `geopandas`, `rioxarray`, `pyogrio` — the raster/vector I/O stack |
| `viz` | `matplotlib`, `folium`, `plotly`, `seaborn`, `scikit-learn` |
| `train` | `torch`, `torchvision`, `timm`, `segmentation-models-pytorch`, `albumentations`, `optuna`, `mlflow`, `wandb` |
| `serve` | `fastapi`, `uvicorn`, `onnxruntime`, `onnxsim`, `websockets` |
| `labeling` | `laspy`, `s2sphere`, `scipy`, `Pillow`, `faiss-cpu` |
| `vlm` | `transformers`, `torch`, `open-clip-torch`, `faiss-cpu` |
| `xai` | `shap`, `captum`, `torch`, `matplotlib` |
| `foundation` | `transformers`, `torch`, `timm`, `hdbscan`, `umap-learn`, `faiss-cpu` |
| `advanced` | `transformers`, `torch`, `optuna`, `statsmodels` |
| `cloud` | `boto3`, `sagemaker`, `azure-ai-ml`, `google-cloud-aiplatform` |
| `edge` | `onnxruntime`, `onnxsim` |
| `geo3d` | `laspy`, `scipy`, `pandas` — point cloud / 3D |
| `monitoring` | `scipy`, `matplotlib`, `requests` — drift detection |
| `enterprise` | `cryptography`, `passlib`, `python-jose`, `sqlalchemy` |
| `dev` | `pytest`, `ruff`, `mypy`, `black`, `pre-commit` |

Core dependencies (always installed): `pygeofetch>=2.4.0` (the data
layer), `pystac`/`pystac-client`/`planetary-computer` (STAC fallback),
`pydantic`, `click`, `pyyaml`, `httpx`, `tenacity`, `numpy`, `rich`,
`shapely`, `pyproj`. Requires Python 3.10+.

See [Installation](installation.md) for verified system requirements and troubleshooting.

## 🖥️ Quick Start

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
results = client.search(
    bbox=(-74.1, 40.6, -73.7, 40.9), date_range=("2024-01-01", "2024-04-30"), cloud_cover_max=15,
)
scene = client.download(results[:1], output_dir="./data")[0]
```

```bash
pygeovision channel land_cover --bbox -74.1 40.6 -73.7 40.9 --date 2024-01
```

See [Quick Start](quickstart.md) for the full walkthrough.

## 📋 Documentation

- [Installation](installation.md) — detailed setup, extras, verified requirements
- [Quick Start](quickstart.md) — first pipeline run in 5 minutes
- [Architecture](architecture.md) — the complete, honest picture of what's verified, what's fixed, and what isn't yet
- [API Reference](api/index.md) — every module, with real, per-page verification status
- [Task Pipelines](api/task_pipelines.md) — all 51 pipelines individually documented
- [Domain Examples](examples/index.md) — end-to-end workflows by application area
- [FAQ](faq.md)

## 🤝 Contributing

Contributions are welcome — bug fixes, new pipeline implementations
(particularly for the 12 pipelines currently honest about not being
implemented), model verification (the 36 network-blocked and 26
forward-pass-unverified models in the registry are the highest-value
targets), and documentation improvements. See the
[Contributing guide](contributing.md).

## 📄 License

PyGeoVision is free and open source software, licensed under the MIT License.

---

*This documentation follows a verification-first standard: every claim is checked against the real, installed code or explicitly labeled unverified — modeled on the audit discipline described in [Architecture](architecture.md), not on marketing copy.*
