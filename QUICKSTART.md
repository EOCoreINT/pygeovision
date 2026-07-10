# PyGeoVision v2.0 — Quick Start

**5 minutes to your first geospatial AI result.**

---

## 1. Install

```bash
# Recommended — AI inference + satellite data
pip install "pygeovision[geo,train]"

# Add foundation models (DINOv3, Prithvi-EO-2.0)
pip install "pygeovision[geo,train,foundation]"

# Everything
pip install "pygeovision[all]"
```

## 2. Verify

```bash
pygeovision status
# ✓ PyGeoVision 2.1.5
# ✓ PyGeoFetch 1.0.0  (22 providers, 10 open access)
# ✓ Model registry    119 architectures
# ✓ Dataset registry  503 datasets
# ✓ Tests             580 passing
```

## 3. Inference on a local GeoTIFF (no credentials needed)

```python
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()

model  = get_model("segformer-b2", num_classes=2, pretrained=True)
inf    = TiledInference(model, chip_size=512, overlap=64)
result = inf.infer("my_scene.tif", "prediction.tif")
# → {n_chips: 48, chips_per_second: 44.7}
```

## 4. Preprocess a scene before inference

```python
# Stack 6 Sentinel-2 bands → clip to study area → normalise → validate
result = client.prepare_for_ai(
    "./downloads/S2C_20240628/",        # scene directory
    stack_bands = ["B02","B03","B04","B08","B11","B12"],
    bbox        = (-74.1, 40.6, -73.7, 40.9),
    scl_path    = "./downloads/S2C_20240628/SCL.tif",
    normalise   = "scale_factor",        # ÷10000 → [0,1]
    model_type  = "segmentation",
    output_path = "ready.tif",
)
arr = result["array"]   # float32 (6, H, W) — validated, AI-ready
```

## 5. Compute spectral indices

```python
# From a stacked 6-band GeoTIFF [B02, B03, B04, B08, B11, B12]
ndvi = client.indices.ndvi("ready.tif", red_band=3, nir_band=4,
                            output_path="ndvi.tif")
nbr  = client.indices.nbr("ready.tif",  nir_band=4, swir2_band=6,
                            output_path="nbr.tif")

# All 22 indices at once
results = client.indices.compute_all("ready.tif", output_dir="./indices/")
```

## 6. Run a foundation model

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone
from pygeovision.models.foundation.prithvi import PrithviTasks

# DINOv3 — satellite pre-trained (SAT-493M)
backbone = DINOv3Backbone("dinov3_vitl16_sat")
emb      = backbone.extract_embeddings("ready.tif")   # (1, 1024)

# Prithvi-EO-2.0 — land cover
tasks = PrithviTasks("prithvi_eo_2_0")
lc    = tasks.land_cover("ready.tif", source="sentinel2")
```

## 7. Postprocess predictions

```python
# Vectorise prediction raster → GeoJSON
client.postprocess.vectorise(
    "prediction.tif",
    "buildings.geojson",
    target_class = 1,
    min_area_m2  = 25.0,
)

# Per-class statistics
stats = client.postprocess.class_statistics("prediction.tif")
# {0: {"pixels": 1234, "area_ha": 0.5, "pct": 12.3}, ...}

# Accuracy vs reference
report = client.postprocess.accuracy_assessment(
    "prediction.tif", "reference.tif")
# {"overall_accuracy": 0.934, "kappa": 0.912, "mean_iou": 0.876}
```

## 8. Add satellite data (optional — PyGeoFetch)

```python
# 10 providers work with no credentials (planetary_computer, aws_earth, element84 …)
results   = client.search(
    bbox            = (-74.1, 40.6, -73.7, 40.9),
    date_range      = ("2024-06-01", "2024-06-30"),
    providers       = ["planetary_computer"],
    cloud_cover_max = 10,
)
downloads = client.download(
    results[:1],
    output_dir   = "./data/",
    bands        = ["B02","B03","B04","B08","B11","B12"],
    post_process = ["reproject:EPSG:4326", "cog"],
)
```

## 9. Run an end-to-end pipeline

```bash
# CLI
pygeovision pipeline building_footprints \
    --bbox "-0.15,51.47,-0.10,51.52" \
    --date 2024-06 \
    --output ./results/

# Python
result = client.pipeline("building_footprints",
    bbox=(-0.15,51.47,-0.10,51.52), date="2024-06")
```

## 10. Open the notebooks

```bash
cd projects/
jupyter lab
```

All 25 notebooks run without GPU and without satellite credentials.

---

## Key Concepts

| Concept | Description |
|---|---|
| `client.validator` | **Mandatory gate** — every dataset is validated before any model |
| `client.prepare_for_ai()` | Stack → clip → mask → normalise → validate in one call |
| `client.preprocess` | Own rasterio-based preprocessing (stack, clip, SCL mask, normalise, resample) |
| `client.indices` | 22 spectral indices (NDVI, EVI, NBR, TCT, PCA …) |
| `client.postprocess` | Vectorise, sieve, smooth, accuracy assessment, COG, zonal stats |
| `client.sar` | SAR processing proxy (requires PyGeoFetch v2.0) |
| `client._pgf_bridge` | PyGeoFetch Python API bridge (search, download, batch process) |

## Common Errors

| Error | Fix |
|---|---|
| `TypeError: unexpected keyword argument 'limit'` | Fixed in v2.0 — `limit=` is now an alias for `max_results=` |
| `TypeError: unexpected keyword argument 'bands'` | Fixed in v2.0 — `bands=` is now supported in `download()` |
| `25 validation errors for SatelliteData` | Fixed in v2.0 — `SatelliteAsset` is now built correctly from cache |
| `expected input to have 4 channels, got 1` | Fixed in v2.0 — ChangeFormer now handles channel mismatches via `_fix_channels()` |
| `'NoneType' cannot be interpreted as an integer` | Fixed in v2.0 — Prithvi config integers are now patched before load |
| `PyGeoFetch not installed` | `pip install PyGeoFetch` — or pass local GeoTIFFs directly to AI APIs |
