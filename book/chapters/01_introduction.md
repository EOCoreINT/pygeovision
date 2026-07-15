# Chapter 1: Introduction to PyGeoVision

> *"The gap between satellite data and actionable intelligence used to take weeks.
> PyGeoVision closes it in minutes."*

## 1.1 What is PyGeoVision?

PyGeoVision is a production-ready Python platform that unifies satellite data
acquisition and geospatial AI in one coherent API. It answers the question that
every remote sensing practitioner has faced: *how do I get from a raw satellite
scene to an AI-generated result without writing hundreds of lines of glue code?*

The answer is a three-layer architecture:

```
Layer 1 — Data         PyGeoFetch: 22+ satellite providers, unified search/download
Layer 2 — AI           119 model architectures, Prithvi, DINOv3, ChangeFormer
Layer 3 — Intelligence GeoAgent: natural language → complete pipeline
```

You interact primarily with the top-level `PyGeoVision` client, which delegates
to these layers automatically:

```python
import pygeovision as pgv

# One client — three layers
client = pgv.PyGeoVision()

# Layer 1: search 22 providers simultaneously
results = client.search(
    bbox=(-0.30, 5.50, -0.05, 5.70),        # Accra, Ghana
    date_range=("2026-06-01", "2026-06-30"),
    cloud_cover_max=20,
)

# Layer 2: preprocess + AI in one call
ready = client.prepare_for_ai(results[0].path, model_type="foundation")

# Layer 3: natural language query (no tool names needed)
from pygeovision.agent import GeoAgent
agent = GeoAgent(client)
agent.set_context(bbox=(-0.30, 5.50, -0.05, 5.70), date="2026-06")
trace = agent.run("Map flood inundation in the Odaw River Basin")
print(trace.final_output)   # → ./results/flood_mask.geojson
```

## 1.2 The PyGeoVision Ecosystem

PyGeoVision is the bridge between two exceptional open-source packages:

### PyGeoFetch — Universal Satellite Data Pipeline

[PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch) handles everything
before data touches your disk:

- Authenticated search across 22+ providers (Sentinel, Landsat, Planet, Maxar,
  Copernicus, USGS, NASA, JAXA, and more)
- Parallel downloads with checksum verification and resume support
- Post-processing chains: unzip → reproject → compress → Cloud Optimised GeoTIFF
- YAML pipeline orchestration with cron scheduling

PyGeoVision wraps PyGeoFetch behind `client.search()` and `client.download()`,
adding validation, the mandatory pipeline gate (§1.4), and direct handoff to AI.

### GeoAI — Artificial Intelligence for Geospatial Data

[GeoAI](https://opengeoai.org) (Qiusheng Wu et al., 2026) provides the
complete AI inference stack: segmentation, detection, classification, change
detection, foundation models, cloud masking, super-resolution, and ONNX export.

PyGeoVision exposes GeoAI as optional plugin (`pip install "pygeovision[geoai]"`).
The core platform includes its own 119-model registry and does not require GeoAI
for any of its standard workflows.

### PyGeoVision — The Integration Layer

PyGeoVision adds what neither package provides alone:

| Feature | PyGeoFetch | GeoAI | PyGeoVision |
|---------|-----------|-------|-------------|
| Data search/download | ✅ | ❌ | ✅ (via PyGeoFetch) |
| AI inference | ❌ | ✅ | ✅ (own stack + GeoAI plugin) |
| End-to-end pipelines | ❌ | ❌ | ✅ 10 named pipelines |
| SAR preprocessing | ❌ | ❌ | ✅ S0-S9 with 3 bug-fixes |
| InSAR chain | ❌ | ❌ | ✅ interferogram → report |
| Natural language AI | ❌ | ❌ | ✅ GeoAgent (heuristic + LLM) |
| Interactive visualization | ❌ | limited | ✅ Map, RasterViewer, etc. |
| Enterprise (RBAC, audit) | ❌ | ❌ | ✅ |

## 1.3 Why PyGeoVision?

### The fragmentation problem

A typical geospatial AI workflow before PyGeoVision looked like this:

1. Search EO Browser or Copernicus Hub manually
2. Download and unzip a 2 GB Sentinel-2 archive
3. Reproject with GDAL command-line tools
4. Stack bands with rasterio
5. Apply cloud mask with Sen2Cor
6. Normalise values for the model
7. Run inference with a model from three different GitHub repos
8. Post-process with shapely + fiona
9. Export a COG manually

Each step has its own API, its own failure modes, and its own documentation.

PyGeoVision collapses this to:

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

results   = client.search(bbox=bbox, date_range=dates, cloud_cover_max=10)
downloads = client.download(results[:1], bands=["B02","B03","B04","B08","B11","B12"])
ready     = client.prepare_for_ai(downloads[0].path)
result    = client.pipeline("building_footprints", bbox=bbox, date="2026-06")
```

### Comparison with alternatives

| Platform | Data access | AI models | SAR | InSAR | Agent | Open source |
|----------|------------|-----------|-----|-------|-------|-------------|
| **PyGeoVision** | 22+ providers | 119 | ✅ | ✅ | ✅ | ✅ Apache 2 |
| TorchGeo | Limited | Many | ❌ | ❌ | ❌ | ✅ |
| TerraTorch | ❌ | Foundation only | ❌ | ❌ | ❌ | ✅ |
| EO-learn | Some | Few | ❌ | ❌ | ❌ | ✅ |
| GEE Python | Google only | Limited | ❌ | ❌ | ❌ | ❌ |
| ArcGIS Pro | Limited | Limited | Limited | ❌ | Limited | ❌ |

## 1.4 The Mandatory Pipeline Gate

Every PyGeoVision workflow passes through a validation gate before data
touches an AI model. This prevents the most common class of errors in
production geospatial AI: feeding the wrong data to a model.

```
Raw download
    │
    ▼
validate_georeference()     ← BUG 1 fix: detects identity-transform CRS corruption
    │
    ▼
check_download_complete()   ← BUG 2 fix: rejects partial downloads before AI
    │
    ▼
prepare_for_ai()            ← BUG 3 fix: CRS-aware bbox before clip
    │   stack bands
    │   cloud mask (SCL)
    │   normalise (scale_factor=10000)
    │   validate shape for model type
    │
    ▼
AI model
```

The three bug-fixes embedded in this gate address real production failures
discovered during development:

- **BUG 1:** After PyGeoFetch reproject, some scenes have a pixel_width of 1.0
  and an origin at (0, 0) instead of valid UTM metres — a silent CRS corruption
  that causes models to silently produce incorrect results.

- **BUG 2:** PyGeoFetch's chunked download can produce a syntactically valid
  but spatially incomplete GeoTIFF. Without a tile-read verification, the
  subsequent despeckle step on a 0-tile SAR file crashes mid-pipeline.

- **BUG 3:** `rasterio.mask` requires the clip geometry to be in the raster's
  CRS. Passing a WGS84 bbox to a UTM raster silently produces an empty output.

## 1.5 Installation

### Requirements

- Python 3.10 or later
- pip 23+
- 4 GB RAM minimum; 16 GB for training

### Install options

```bash
# Minimal — search, download, basic AI
pip install pygeovision

# Recommended — full geospatial stack
pip install "pygeovision[geo]"

# With visualization
pip install "pygeovision[geo,viz]"

# With AI training (PyTorch)
pip install "pygeovision[geo,train]"

# With InSAR processing
pip install "pygeovision[geo,insar]"

# With enterprise features (RBAC, audit, compliance)
pip install "pygeovision[enterprise]"

# Everything
pip install "pygeovision[all]"
```

### Verify installation

```python
import pygeovision as pgv
print(pgv.__version__)   # 2.1.6

# Check available components
client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.1.6, providers=22, models=119, datasets=503)
```

## 1.6 Quick Start: Your First Analysis

This five-minute workflow searches Sentinel-2 imagery over London, downloads
one scene, computes NDVI, and visualises the result.

```python
# ─── 1. Import and initialise ──────────────────────────────────────────────
import pygeovision as pgv
from pygeovision.viz import RasterViewer

client = pgv.PyGeoVision()

# ─── 2. Define study area (London, UK) ────────────────────────────────────
BBOX       = (-0.15, 51.47, -0.10, 51.52)   # [lon_min, lat_min, lon_max, lat_max]
DATE_RANGE = ("2026-06-01", "2026-06-30")
BANDS      = ["B02", "B03", "B04", "B08", "B8A", "B11"]   # Sentinel-2 HLS subset

# ─── 3. Search open providers (no credentials needed) ─────────────────────
results = client.search(
    bbox            = BBOX,
    date_range      = DATE_RANGE,
    providers       = ["planetary_computer"],   # Microsoft, free tier
    cloud_cover_max = 15,
)
print(f"Found {len(results)} scenes")
for r in results[:5]:
    print(f"  {r.provider:<22}  {r.datetime[:10]}  cloud={r.cloud_cover:.1f}%")
# Found 4 scenes
#   planetary_computer      2026-06-08  cloud=3.2%
#   planetary_computer      2026-06-18  cloud=8.1%
#   ...

# ─── 4. Download the clearest scene ───────────────────────────────────────
downloads = client.download(
    results[:1],
    output_dir   = "./data/london/",
    bands        = BANDS,
    post_process = ["reproject:EPSG:32630", "cog"],   # UTM 30N + COG
)
scene_path = downloads[0].path
print(f"Downloaded: {scene_path}  ({downloads[0].bytes_downloaded/1e6:.1f} MB)")
# Downloaded: ./data/london/S2_20260608_B*.tif  (156.4 MB)

# ─── 5. Prepare for AI (mandatory pipeline gate) ──────────────────────────
ready = client.prepare_for_ai(
    scene_path,
    stack_bands  = BANDS,
    bbox         = BBOX,
    bbox_crs     = "EPSG:32630",
    normalise    = "scale_factor",
    scale_factor = 10000.0,
    model_type   = "segmentation",
    output_path  = "./data/london/london_prepared.tif",
)
print(f"Shape: {ready['shape']}  Range: [{ready['array'].min():.3f}, {ready['array'].max():.3f}]")
# Shape: (6, 512, 512)  Range: [0.000, 0.981]

# ─── 6. Compute NDVI and visualise ────────────────────────────────────────
import numpy as np

arr  = ready["array"]   # (6, H, W) normalised float32
nir  = arr[3]           # Band 8 (NIR)
red  = arr[2]           # Band 4 (Red)
ndvi = (nir - red) / (nir + red + 1e-8)
print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
print(f"Mean NDVI (parks/vegetation): {ndvi.mean():.3f}")
# NDVI range: -0.12 to 0.73
# Mean NDVI (parks/vegetation): 0.31

rv = RasterViewer("./data/london/london_prepared.tif")
rv.ndvi(nir_band=3, red_band=2).export("./data/london/ndvi_london.png")
rv.rgb(red=2, green=1, blue=0).show()   # True-colour composite

# ─── 7. Run a full AI pipeline with one call ──────────────────────────────
result = client.pipeline(
    "building_footprints",
    bbox       = BBOX,
    date       = "2026-06",
    output_dir = "./data/london/",
)
print(f"Buildings detected: {result.stats.get('n_buildings', 'N/A')}")
print(f"Output: {result.output_path}")
# Buildings detected: 2,847
# Output: ./data/london/buildings.geojson
```

**Expected time:** ~3 minutes (download-dominated).  
**Expected output:** `ndvi_london.png` and `buildings.geojson`.

## 1.7 Key Concepts

| Term | Meaning |
|------|---------|
| **bbox** | Bounding box `[lon_min, lat_min, lon_max, lat_max]` in WGS84 |
| **COG** | Cloud Optimised GeoTIFF — spatially indexed for web streaming |
| **HLS** | Harmonised Landsat Sentinel — Prithvi's expected 6-band input format |
| **LOS** | Line-of-Sight — the direction along the satellite radar beam |
| **SAR** | Synthetic Aperture Radar — cloud-independent microwave imaging |
| **InSAR** | Interferometric SAR — detects millimetre-scale ground deformation |
| **SCL** | Scene Classification Layer (Sentinel-2) — cloud/shadow mask |
| **SLC** | Single Look Complex — SAR data with phase information (InSAR input) |

## 1.8 Chapter Summary

- PyGeoVision unifies satellite data (22+ providers) and AI (119 architectures)
  behind a single Python API.
- The mandatory pipeline gate catches the three most common production failures
  before they reach the model.
- The GeoAgent accepts natural-language queries and autonomously executes the
  complete workflow — sensor choice, task type, and tool sequence all decided
  from the problem description.
- Installation: `pip install "pygeovision[all]"`.

## Exercises

1. Install PyGeoVision and run the London quick-start example.
2. Change the study area to your city. What is the mean NDVI?
3. Try `cloud_cover_max=5` — how many scenes are available for your area?
4. Change the pipeline from `building_footprints` to `water_bodies`.
   How much water area does your study area contain?

## Further Reading

- PyGeoFetch documentation: https://appiahkubis14.github.io/pygeofetch-docs/
- GeoAI paper: Wu (2026), *Journal of Open Source Software*. DOI: 10.21105/joss.09605
- Sentinel-2 product specification: ESA S2-PDGS-TAS-DI-PSD
- Prithvi-EO-2.0: Jakubik et al. (2023), arXiv:2310.18660
