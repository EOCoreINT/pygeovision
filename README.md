<div align="center">

# PyGeoVision

[![image](https://img.shields.io/pypi/v/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![image](https://img.shields.io/pypi/l/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![image](https://img.shields.io/pypi/pyversions/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![Tests](https://img.shields.io/badge/Tests-864_passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](https://appiahkubis14.github.io/pygeovision-docs/)
[![PyGeoFetch](https://img.shields.io/badge/PyGeoFetch-22_providers-f59e0b?style=flat-square)](https://appiahkubis14.github.io/pygeofetch-docs/)
[![Models](https://img.shields.io/badge/Models-119%2B_native-a855f7?style=flat-square)](#-ai-inference--fully-native)

**A production-ready, fully self-contained Python platform unifying satellite data acquisition and geospatial AI —
one native stack, no external AI-platform dependency.**

</div>

---

## 📖 Introduction

PyGeoVision is a **world-class geospatial AI platform** built on two layers, both maintained inside this package:

- **Data layer** — delegates satellite search, download, authentication, caching, and pipeline orchestration to [PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch), giving access to 22+ providers including Sentinel, Landsat, Planet, Maxar, Copernicus, USGS, and more.
- **AI layer** — implemented natively inside PyGeoVision itself: segmentation (SAM), detection (YOLO), change detection (ChangeFormer), classification (CLIP + ESA WorldCover), a 119+ model architecture registry, foundation models (Prithvi-EO-2.0, DINOv3), auto-labeling, explainability, drift monitoring, and edge/cloud deployment. Nothing here is a wrapper around a separate AI platform — every prediction path runs PyGeoVision's own code.

PyGeoVision's design principle is **own the whole stack, end to end**: one dependency-light data layer, one native AI layer, wired together with 51 built-in domain pipelines, a full CLI, an inference server, experiment tracking, distributed training, automated labeling, and YAML pipeline orchestration.

The package provides six core capabilities:

1. Authenticated search and download of satellite imagery from 22+ providers with caching, parallel downloads, and post-processing chains.
2. End-to-end pipelines that wire PyGeoFetch data directly into native AI inference with a single function call.
3. A complete AI training stack with a 119+ architecture registry, specialist losses, and distributed GPU support.
4. Automated dataset labeling from 7 sources including OSM, Microsoft, Google, ESA WorldCover, SAM, and foundation models.
5. A full CLI (`pygeovision`) covering data, AI, models, pipelines, and inference serving.
6. YAML pipeline orchestration for scheduled, repeatable geospatial workflows.

---

## 📝 Statement of Need

Applying AI to geospatial data requires navigating fragmented ecosystems — separate tools for data acquisition, preprocessing, model training, and inference — leading to steep learning curves, brittle pipelines, and reproducibility challenges. Existing packages like TorchGeo and TerraTorch provide excellent foundational tools but leave the data acquisition layer largely unsolved.

PyGeoVision fills this gap by providing a unified, high-level interface that:

- Gives **geospatial researchers** a single import to go from satellite search to AI inference, with no separate AI-platform installation or licensing to manage.
- Gives **AI practitioners** streamlined access to 22 satellite data providers without managing APIs, authentication, and file formats manually.
- Gives **organizations** a production-ready platform with CLI tooling, a REST/WebSocket inference server, YAML pipelines, scheduling, and experiment tracking.

With 51 built-in end-to-end pipelines covering building footprints, change detection, land cover, water bodies, solar detection, crop monitoring, disaster assessment, deforestation, urban growth, carbon estimation, and more, PyGeoVision dramatically reduces the time from raw satellite imagery to actionable geospatial intelligence.

---

### 🛰️ Satellite Data — 22 Providers via PyGeoFetch
- Unified search and download across Sentinel, Landsat, Planet, Maxar, Airbus, USGS, Copernicus, NASA, JAXA, and more
- Secure credential management via system keyring (API keys, OAuth2, user/password)
- Parallel downloads with checksum verification, resume support, and bandwidth throttling
- Preprocessing — Full optical and SAR preprocessing pipelines: atmospheric correction (DOS1/DOS2/Sen2Cor), cloud masking (SCL/FMask), topographic correction, pan-sharpening, mosaicking, multi-date compositing, and re-projection to any CRS
- Post-processing chains: unzip → reproject → compress → NDVI/NDWI → Cloud Optimized GeoTIFF
- YAML pipeline orchestration with cron scheduling

### 🤖 AI Inference — Fully Native
- **Segmentation** (`client.segmentation`) — buildings, water (NDWI), general SAM auto-segmentation, custom models
- **Detection** (`client.detection`) — generic / ships / cars via native YOLO, custom models
- **Classification** (`client.classification`) — CLIP zero-shot scene classification, ESA WorldCover land cover
- **Change detection** (`client.change`) — ChangeFormer bi-temporal transformer, with a dependency-free spectral-diff fallback
- **Foundation models** — NASA Prithvi-EO-2.0 (land cover, crop mapping, flood/burn-scar detection, time series), DINOv3 (features, embeddings, zero-shot text, canopy height via CHMv2), SAM, Tessera satellite embeddings
- **Explainability** (`client.xai`) — GradCAM, GradCAM++, uncertainty estimation, attention maps, SHAP
- **Monitoring** (`client.monitoring`) — drift detection, performance tracking, alerting
- **Edge / cloud deployment** (`client.edge` / `client.cloud`) — ONNX Runtime, NVIDIA Jetson, AWS SageMaker, Azure ML, GCP Vertex AI
- **Inference server** — FastAPI REST + WebSocket serving with batch inference, model registry, and request metrics
- Cloud masking, super-resolution, ONNX export

### ⚙️ End-to-End Pipelines (51)
| Pipeline | Description |
|----------|-------------|
| `building_footprints` | Sentinel-2 / NAIP → SAM segmentation → GeoTIFF/vector |
| `change_detection` | Bi-temporal Sentinel-2 → ChangeFormer → change mask |
| `land_cover` | Sentinel-2 → ESA WorldCover / Prithvi → classification map |
| `water_bodies` | Sentinel-2 → NDWI segmentation → water mask |
| `solar_detection` | NAIP / Sentinel-2 → native detection → GeoTIFF |
| `crop_monitoring` | Seasonal Sentinel-2 stack → crop type map |
| `disaster_assessment` | Post-event imagery → change detection → damage assessment |
| `deforestation` | Bi-temporal Landsat/S2 → ChangeFormer → forest loss mask |
| `urban_growth` | Bi-temporal Landsat → change detection → urban expansion map |
| `carbon_estimation` | Sentinel-2 NDVI → allometric AGB formula → carbon stock estimate |

*(41 additional domain pipelines are available — run `pygeovision pipeline list` or `client.pipeline("list")` for the full catalogue.)*

### 🧠 Own AI Training Stack
- 119+ registered model architectures: U-Net, SegFormer, DeepLabV3+, FCOS, RetinaNet, ViT, ChangeFormer, ESRGAN, YOLOv8/v9, foundation-model backbones, and more
- `GeoTrainer` with specialist losses (Dice, Focal, Tversky, Boundary-Aware, Lovász, OHEM, Combo, Class-Balanced)
- Distributed multi-GPU training, mixed precision, gradient accumulation
- `TiledInference` with Gaussian blending for large-scene inference
- `ExperimentTracker` and `DriftDetector` for production monitoring

### 🏷️ Automated Labeling (7 Sources)
OpenStreetMap · Microsoft Global Buildings · Google Open Buildings · ESA WorldCover · Google Dynamic World · SAM auto-labeling · Foundation model labeling

### 🛰️ SAR & InSAR — GRD and SLC Preprocessing to Displacement Maps
A full Sentinel-1 processing chain, natively implemented — not a SNAP wrapper:

- **GRD preprocessing (`client.sar`, `pygeovision.data.processors.sar`)** — a documented 10-stage pipeline (S0–S9): download-completeness verification → georeference validation with automatic GCP/raw-reference repair → thermal noise removal → radiometric calibration (sigma-naught) → terrain correction → despeckle (boxcar, enhanced Lee, or a genuine directional-window **refined Lee** filter) → dB conversion → AI-ready [0,1] normalisation → CRS-aware bbox clipping.
- **SAR → foundation-model bridging (`models.adapters.sar_channel_manager`)** — maps 2-band VV/VH SAR to the 6-channel HLS format Prithvi expects (physics-guided, ratio-based, or replicate mappings), and to 3-channel pseudo-RGB for DINOv3, with sub-pixel co-registration for change detection.
- **SAR-adapted foundation models** — `SARPrithviAdapter` (zero-shot flood detection + fine-tuning config for TerraTorch) and `SARDINOv3Adapter` (SAR-domain feature extraction with speckle-aware augmentation).
- **InSAR (`pygeovision.insar`, `InSARProcessor`)** — GRD amplitude-proxy interferogram generation (Goldstein filtering), coherence estimation, phase-to-displacement conversion, and an interpretation layer that classifies subsidence/uplift zones and flags anomalies from a displacement map, end to end in three method calls.
- **Full SLC InSAR (`pygeovision.insar.slc`)** — orchestrates true phase-based InSAR from Sentinel-1 SLC products: SNAP co-registration → interferogram → Goldstein filter → SNAPHU phase unwrapping → LOS displacement → terrain correction, for centimetre-precision deformation monitoring (requires ESA SNAP + snaphu).

---

## 📦 Installation

```bash
# Core — data + basic inference
pip install pygeovision

# + Geospatial processing (rasterio, geopandas, rioxarray)
pip install "pygeovision[geo]"

# + Training stack (PyTorch, SMP, transformers, timm)
pip install "pygeovision[train]"

# + Foundation models (DINOv3, Prithvi-EO-2.0)
pip install "pygeovision[foundation]"

# + Vision-language models (CLIP, Moondream)
pip install "pygeovision[vlm]"

# + Auto-labeling sources
pip install "pygeovision[labeling]"

# + InSAR (GRD amplitude-proxy chain; SLC InSAR additionally needs ESA SNAP + snaphu)
pip install "pygeovision[insar]"

# + Explainability / XAI
pip install "pygeovision[xai]"

# + Time-series analysis
pip install "pygeovision[timeseries]"

# + Inference server (FastAPI, uvicorn, websockets)
pip install "pygeovision[serve]"

# + Edge deployment (ONNX Runtime)
pip install "pygeovision[edge]"

# + Cloud deployment (AWS/Azure/GCP SDKs)
pip install "pygeovision[cloud]"

# + Enterprise (RBAC, SSO, audit logging)
pip install "pygeovision[enterprise]"

# + Everything (excluding cloud/edge — install those explicitly)
pip install "pygeovision[all]"
```

**Requirements:** Python 3.10+ · PyGeoFetch · PyTorch 2.0+ (only required for training/inference on deep models — search, download, NDWI/NDVI segmentation, and land-cover labeling work without it)

---
## ⚡ Quick Start

### Land Cover with Prithvi-EO-2.0
Ethiopian Highlands | Sentinel-2 | Prithvi 600M

```python
import warnings; warnings.filterwarnings('ignore')
import pathlib, json
import numpy as np
import matplotlib.pyplot as plt
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()
print(client)

from pygeovision.models.foundation.prithvi import PrithviTasks

BBOX       = (38.6, 8.9, 38.95, 9.15)
DATE_RANGE = ('2024-01-01', '2024-04-30')
PROVIDERS  = ['planetary_computer']

DATA_DIR = pathlib.Path('./results/notebook')
DATA_DIR.mkdir(parents=True, exist_ok=True)

results = client.search(
    bbox            = BBOX,
    date_range      = DATE_RANGE,
    providers       = PROVIDERS,
    cloud_cover_max = 15,
)
print(f"Found {len(results)} scenes")
for r in results[:5]:
    print(f"  {r.provider:<22} {r.datetime[:10]}  cloud={r.cloud_cover:.1f}%  {r.id[:40]}")

BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

downloads = client.download(
    results[:1],
    output_dir   = str(DATA_DIR),
    bands        = BANDS,
    post_process = ['reproject:EPSG:32637', 'cog'],
)
scene_path = downloads[0].path if downloads and downloads[0].success else None
scl_cands  = list(DATA_DIR.rglob('*SCL*.tif')) + list(DATA_DIR.rglob('*scl*.tif'))
scl_path   = str(scl_cands[0]) if scl_cands else None
if scene_path:
    d = downloads[0]
    print(f"Downloaded   : {scene_path}")
    print(f"Size         : {d.bytes_downloaded/1024/1024:.1f} MB")
    print(f"SCL mask     : {scl_path}")
else:
    print("Download failed or scene unavailable — check provider availability")

if scene_path:
    raw_report = client.validator.validate(
        str(scene_path),
        required_bands = 6,
        value_range    = (0, 65535),
    )
    print("Raw data validation:")
    print(raw_report.summary())
    if not raw_report.passed:
        print("WARNING: Issues found — preprocessing will auto-fix")

PREPROCESSED = DATA_DIR / 'ethiopia_prithvi_ready.tif'

if scene_path:
    ready = client.prepare_for_ai(
        str(scene_path),
        stack_bands  = BANDS,
        bbox         = BBOX,
        scl_path         = scl_path,
        scl_keep_classes = [4,5,6],
        normalise    = 'scale_factor',
        scale_factor = 10000.0,
        model_type   = 'foundation',
        output_path  = str(PREPROCESSED),
    )
    print("Preprocessing steps :", ready['steps'])
    print("Output shape (C,H,W):", ready['shape'])
    print("Resolution          :", ready['resolution_m'], "m")
    print("Validation          :", "PASSED" if ready['report'] and ready['report'].passed else "FIXED (auto)")
    arr = ready['array']
    print(f"Value range         : {arr.min():.4f} → {arr.max():.4f}")
    print(f"NaN count           : {np.isnan(arr).sum()}")
else:
    print("No scene available — cannot preprocess")
    ready = None

# Prithvi-EO-2.0 land cover — automatically applies map_bands + normalise_hls
PREDICTION = DATA_DIR / 'land_cover.tif'
CLASS_NAMES = ['Tree cover','Shrubland','Grassland','Cropland',
               'Built-up','Bare/sparse','Snow/ice','Water','Wetland']
if PREPROCESSED.exists():
    tasks = PrithviTasks('prithvi_eo_2_0')
    tasks.land_cover(str(PREPROCESSED), source='sentinel2', output_path=str(PREDICTION))
    print(f"Land cover map saved: {PREDICTION}")

if PREDICTION.exists():
    pred_report = client.validator.validate(str(PREDICTION))
    print("Prediction validation:", "PASSED" if pred_report.passed else f"FIXED — {pred_report.errors}")
    print(pred_report.summary())

SIEVED = DATA_DIR / 'lc_sieved.tif'
COG_OUT = DATA_DIR / 'lc_cog.tif'
if PREDICTION.exists():
    client.postprocess.sieve_filter(str(PREDICTION), min_pixels=50, output_path=str(SIEVED))
    stats = client.postprocess.class_statistics(str(SIEVED))
    client.postprocess.to_cog(str(SIEVED), str(COG_OUT))
    client.postprocess.generate_report(str(SIEVED), str(DATA_DIR/'lc_report.html'))
    print(f"{'Cls':<4} {'Name':<22} {'Area(ha)':>10} {'Cover':>8}")
    print('-'*48)
    for i, name in enumerate(CLASS_NAMES):
        inf = stats.get(i, {})
        print(f"{i:<4} {name:<22} {inf.get('area_ha',0):>10.1f} {inf.get('pct',0):>7.1f}%")
```

### Native segmentation, detection, and change in three lines

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

client.segmentation.buildings("scene.tif", output_path="buildings.tif")
client.detection.ships("port.tif", output_path="ships.tif")
client.change.detect(before="2020.tif", after="2024.tif", output_path="change.tif")
```

### SAR preprocessing and InSAR displacement

```python
import pygeovision as pgv
from pygeovision.insar import InSARProcessor

client = pgv.PyGeoVision()

# Sentinel-1 GRD → despeckled, calibrated, AI-ready sigma-naught
despeckled = client.sar.despeckle("sentinel1_raw.tif", filter="enhanced_lee")
calibrated = client.sar.calibrate(despeckled, output_type="sigma0", in_db=True)

# Amplitude-proxy InSAR: interferogram → displacement → interpreted report
insar = InSARProcessor(output_dir="./insar/")
result = insar.full_pipeline("pre_event.tif", "post_event.tif", study_area="Jakarta")
print(result.report.summary())
```

---

## 🖥️ Command Line

```bash
# Satellite data
pygeovision data search --bbox -74.1 40.6 -73.7 40.9 --providers planetary_computer
pygeovision data download --bbox -74.1 40.6 -73.7 40.9 --output ./data/

# Native AI
pygeovision ai segment buildings --input scene.tif --output buildings.tif
pygeovision ai detect ships --input port.tif --output ships.tif
pygeovision ai change --before 2020.tif --after 2024.tif --output change.tif
pygeovision ai classify land-cover --input scene.tif --output lc.tif

# Model registry (native, 119+ architectures)
pygeovision models list
pygeovision models info unet_resnet50

# End-to-end domain pipelines (51 built in)
pygeovision pipeline building_footprints --bbox -74.1 40.6 -73.7 40.9 --date 2024-06
pygeovision pipeline list

# System status
pygeovision status
pygeovision doctor
```

Run `pygeovision --help` or `pygeovision <group> --help` for the full command tree — it also covers `label`, `explain`, `monitor`, `edge`, `cloud`, `vlm`, `timeseries`, `validate`, `preprocess`, `indices`, `postprocess`, `benchmark`, `zoo`, and `datasets`. SAR/InSAR preprocessing is available via `client.sar` and `pygeovision.insar` in Python (see [SAR & InSAR](#️-sar--insar--grd-preprocessing-to-displacement-maps) above); it isn't yet exposed as its own top-level CLI group.

---

## 🌐 Inference Server

A FastAPI REST + WebSocket server ships with the package for production serving:

```python
from pygeovision.serving.api import create_app
import uvicorn

app = create_app(auth_keys={"myuser": "myapikey"})
uvicorn.run(app, host="0.0.0.0", port=8080)
```

Endpoints: `POST /predict`, `POST /predict/batch`, `GET /models`, `POST /models/register`, `GET /metrics`, `WS /ws/stream`.

---

## 🧪 Testing

```bash
pip install "pygeovision[dev,train,geo]"
pytest tests/
```

864 tests pass with the full stack (including PyTorch) installed; 620 pass without PyTorch installed, covering everything except deep-model training/inference paths.

---

## 📋 Documentation

Comprehensive documentation is available at **https://appiahkubis14.github.io/pygeovision-docs/**, including:

- Full API reference
- Tutorials and example notebooks
- Pipeline configuration guides
- Contributing guide

---

## 🤝 Contributing

Contributions of all kinds are welcome. See our [contributing guide](CONTRIBUTING.md) for ways to get started.

---

## 📄 License

PyGeoVision is free and open source software, licensed under the [Apache 2.0 License](LICENSE).

---

## Acknowledgements

PyGeoVision's satellite data layer is built on top of **[PyGeoFetch](https://appiahkubis14.github.io/pygeofetch-docs/)** — a universal satellite data pipeline. PyGeoVision delegates all data search, download, authentication, caching, and pipeline orchestration to PyGeoFetch. All AI functionality — segmentation, detection, classification, change detection, foundation models, training, and serving — is implemented natively within PyGeoVision itself.




