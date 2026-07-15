<div align="center">

# PyGeoVision

[![image](https://img.shields.io/pypi/v/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![image](https://img.shields.io/pypi/l/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![image](https://img.shields.io/pypi/pyversions/pygeovision.svg)](https://pypi.python.org/pypi/pygeovision)
[![Tests](https://img.shields.io/badge/Tests-208_passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](https://appiahkubis14.github.io/pygeovision-docs/)
[![PyGeoFetch](https://img.shields.io/badge/PyGeoFetch-22_providers-f59e0b?style=flat-square)](https://appiahkubis14.github.io/pygeofetch-docs/)
[![GeoAI](https://img.shields.io/badge/GeoAI-24_subsystems-a855f7?style=flat-square)](https://opengeoai.org)
[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.09605-blue)](https://doi.org/10.21105/joss.09605)

**A production-ready Python platform unifying satellite data acquisition and geospatial AI —  
bridging [PyGeoFetch](https://appiahkubis14.github.io/pygeofetch-docs/) (22+ providers) and [GeoAI](https://opengeoai.org) (full AI stack) in one coherent API.**

</div>

---

## 📖 Introduction

PyGeoVision is a **world-class geospatial AI platform** that brings together two exceptional open-source packages under a single, unified API. It delegates all satellite data operations to [PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch) — providing access to 22+ providers including Sentinel, Landsat, Planet, Maxar, Copernicus, USGS, and more — and all AI operations to [GeoAI](https://opengeoai.org), exposing its full 24-subsystem stack covering segmentation, detection, classification, change detection, SAM, foundation models, embeddings, cloud masking, super-resolution, and ONNX export.

PyGeoVision's design principle is **integration, not reimplementation**. It is the bridge layer that makes PyGeoFetch and GeoAI work seamlessly together, adding 10 end-to-end pipelines, a full CLI, experiment tracking, distributed training, automated labeling, and YAML pipeline orchestration on top.

The package provides six core capabilities:

1. Authenticated search and download of satellite imagery from 22+ providers with caching, parallel downloads, and post-processing chains.
2. End-to-end pipelines that wire PyGeoFetch data directly into GeoAI inference with a single function call.
3. A complete AI training stack with 14 model architectures, 6 specialist losses, and distributed GPU support.
4. Automated dataset labeling from 7 sources including OSM, Microsoft, Google, ESA WorldCover, SAM, and foundation models.
5. A full CLI (`pygeovision`) covering data, AI, pipelines, and model management.
6. YAML pipeline orchestration for scheduled, repeatable geospatial workflows.

---

## 📝 Statement of Need

Applying AI to geospatial data requires navigating fragmented ecosystems — separate tools for data acquisition, preprocessing, model training, and inference — leading to steep learning curves, brittle pipelines, and reproducibility challenges. Existing packages like TorchGeo and TerraTorch provide excellent foundational tools but leave the data acquisition layer largely unsolved. PyGeoFetch addresses data acquisition comprehensively, and GeoAI addresses the AI layer comprehensively, but combining them into production workflows requires significant integration work.

PyGeoVision fills this gap by providing a unified, high-level interface that:

- Gives **geospatial researchers** a single import to go from satellite search to AI inference.
- Gives **AI practitioners** streamlined access to 22 satellite data providers without managing APIs, authentication, and file formats manually.
- Gives **organizations** a production-ready platform with CLI tooling, YAML pipelines, scheduling, and experiment tracking.

With 10 built-in end-to-end pipelines covering building footprints, change detection, land cover, water bodies, solar detection, crop monitoring, disaster assessment, deforestation, urban growth, and carbon estimation, PyGeoVision dramatically reduces the time from raw satellite imagery to actionable geospatial intelligence.

---

<!-- ## Citations

If you find PyGeoVision useful in your research, please consider citing the following works:

```bibtex
@article{Wu2026geoai,
  author  = {Wu, Qiusheng},
  title   = {GeoAI: A Python package for integrating artificial intelligence with geospatial data analysis and visualization},
  journal = {Journal of Open Source Software},
  year    = {2026},
  volume  = {11},
  number  = {118},
  pages   = {9605},
  doi     = {10.21105/joss.09605}
}
``` -->

---

### 🛰️ Satellite Data — 22 Providers via PyGeoFetch
- Unified search and download across Sentinel, Landsat, Planet, Maxar, Airbus, USGS, Copernicus, NASA, JAXA, and more
- Secure credential management via system keyring (API keys, OAuth2, user/password)
- Parallel downloads with checksum verification, resume support, and bandwidth throttling
- Post-processing chains: unzip → reproject → compress → NDVI/NDWI → Cloud Optimized GeoTIFF
- YAML pipeline orchestration with cron scheduling

### 🤖 AI Inference — 24 Subsystems via GeoAI
- Segmentation: buildings, solar panels, agriculture fields, water bodies, custom models
- Detection: cars, ships, parking spots, natural-language grounded detection (GroundedSAM), RF-DETR
- Classification: scene classification, CLIP zero-shot land cover, batch inference
- Change detection: ChangeSTAR bi-temporal change detection
- Foundation models: NASA Prithvi, SAM, DINOv3, Tessera satellite embeddings
- Cloud masking, super-resolution (ESRGAN), ONNX export, canopy height estimation

### ⚙️ End-to-End Pipelines (10)
| Pipeline | Description |
|----------|-------------|
| `building_footprints` | Sentinel-2 / NAIP → GeoAI BuildingFootprintExtractor → GeoJSON |
| `change_detection` | Bi-temporal Sentinel-2 → ChangeSTAR → change mask |
| `land_cover` | Sentinel-2 → SegFormer / ESA WorldCover → classification map |
| `water_bodies` | Sentinel-2 → NDWI segmentation → water polygons |
| `solar_detection` | NAIP / Sentinel-2 → SolarPanelDetector → GeoJSON |
| `crop_monitoring` | Seasonal Sentinel-2 stack → crop type map |
| `disaster_assessment` | Post-event imagery → Siamese-UNet → damage assessment |
| `deforestation` | Bi-temporal Landsat/S2 → ChangeFormer → forest loss mask |
| `urban_growth` | Bi-temporal Landsat → Siamese-UNet → urban expansion map |
| `carbon_estimation` | Sentinel-2 NDVI → AGB formula → carbon stock estimate |

### 🧠 Own AI Training Stack
- 14 model architectures: U-Net, SegFormer, DeepLabV3+, FCOS, RetinaNet, ViT, ChangeFormer, ESRGAN, and more
- GeoTrainer with 6 specialist losses (Dice, Focal, Tversky, Unified Focal, Weighted CE, Change Detection)
- Distributed multi-GPU training, mixed precision, gradient accumulation
- TiledInference with Gaussian blending for large-scene inference
- ExperimentTracker and DriftDetector for production monitoring

### 🏷️ Automated Labeling (7 Sources)
OpenStreetMap · Microsoft Global Buildings · Google Open Buildings · ESA WorldCover · Google Dynamic World · SAM auto-labeling · Foundation model labeling

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

# + Time-series analysis
pip install "pygeovision[timeseries]"

# + Serving API (FastAPI, uvicorn, websockets)
pip install "pygeovision[serve]"

# + Everything
pip install "pygeovision[all]"
```

**Requirements:** Python 3.10+ · PyGeoFetch · GeoAI (optional) · PyTorch 2.0+

---
## ⚡ Quick Start

# Land Cover with Prithvi-EO-2.0
**NB 03** | Ethiopian Highlands | Sentinel-2 | Prithvi 600M

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

PyGeoVision is built on top of two exceptional open-source projects:

- **[PyGeoFetch](https://appiahkubis14.github.io/pygeofetch-docs/)** — Universal satellite data pipeline. PyGeoVision delegates all data search, download, authentication, caching, and pipeline orchestration to PyGeoFetch.

<!-- - **[GeoAI](https://opengeoai.org)** — Artificial Intelligence for Geospatial Data by [Qiusheng Wu](https://github.com/giswqs) and contributors. PyGeoVision wraps GeoAI for all AI inference, training, and model management. Published in [JOSS 2026](https://doi.org/10.21105/joss.09605). -->