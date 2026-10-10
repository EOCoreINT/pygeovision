# Chapter 3: Data Preprocessing

## 3.1 Why Preprocessing Matters

Raw satellite data is almost never ready for an AI model. A Sentinel-2 L2A
scene contains reflectance values in 12 bands at three different resolutions
(10m, 20m, 60m), encoded as 16-bit integers scaled by 10,000, masked by a
Scene Classification Layer. Preprocessing converts this to a normalised,
validated array that any model can consume.

## 3.2 The prepare_for_ai() Pipeline

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

ready = client.prepare_for_ai(
    input_path   = "./data/S2_20260615.tif",
    stack_bands  = ["B02","B03","B04","B08","B8A","B11"],
    bbox         = (-0.30, 5.50, -0.05, 5.70),
    bbox_crs     = "EPSG:32630",       # BUG 3 fix: must match raster CRS
    scl_path     = "./data/S2_SCL.tif",
    scl_keep_classes = [4, 5, 6],      # vegetation, bare, water
    normalise    = "scale_factor",
    scale_factor = 10000.0,
    model_type   = "segmentation",
    output_path  = "./data/prepared.tif",
)

print("Steps:", ready["steps"])
print("Shape:", ready["shape"])     # (6, 512, 512)
print("Range:", ready["array"].min(), "to", ready["array"].max())
print("Valid:", ready["report"].passed)
```

## 3.3 Cloud Masking

```python
import rasterio
import numpy as np

# SCL classes: 4=vegetation, 5=bare, 6=water, 8/9/10=clouds
with rasterio.open("./data/S2_SCL.tif") as src:
    scl = src.read(1)

clear_mask = np.isin(scl, [4, 5, 6]).astype("uint8")
coverage   = clear_mask.mean() * 100
print(f"Cloud-free coverage: {coverage:.1f}%")

# Apply mask
with rasterio.open("./data/S2_bands.tif") as src:
    data    = src.read().astype("float32")
    profile = src.profile.copy()

data[:, clear_mask == 0] = np.nan

with rasterio.open("./data/cloudmasked.tif", "w", **profile) as dst:
    dst.write(data)
```

## 3.4 Reprojection

```python
from pygeovision.preprocess import Preprocessor
pre = Preprocessor()

# Reproject to UTM
pre.reproject(
    input_path  = "./data/scene.tif",
    output_path = "./data/scene_utm.tif",
    target_crs  = "EPSG:32630",
    resampling  = "bilinear",
)
```

## 3.5 Band Stacking

Sentinel-2 bands come at different resolutions. Stack all at 10m:

```python
pre.stack_bands(
    band_paths = [
        "./data/B02_10m.tif",   # Blue  — 10m native
        "./data/B03_10m.tif",   # Green — 10m native
        "./data/B04_10m.tif",   # Red   — 10m native
        "./data/B08_10m.tif",   # NIR   — 10m native
        "./data/B8A_20m.tif",   # Red-Edge 4 — 20m, upsampled
        "./data/B11_20m.tif",   # SWIR 1     — 20m, upsampled
    ],
    output_path   = "./data/S2_6band_10m.tif",
    target_res_m  = 10,
    resampling    = "bilinear",
    band_names    = ["Blue","Green","Red","NIR","RedEdge4","SWIR1"],
)
```

## 3.6 Mosaicking

```python
pre.mosaic(
    input_paths = [
        "./data/tile_T30PXR.tif",
        "./data/tile_T30PXT.tif",
    ],
    output_path = "./data/mosaic.tif",
    method      = "median",
    nodata      = 0,
)
```

## 3.7 Normalisation Strategies

```python
import numpy as np

# Scale factor (Sentinel-2 standard)
arr_norm = arr.astype("float32") / 10000.0

# Percentile stretch (robust to outliers)
out = np.zeros_like(arr, dtype="float32")
for b in range(arr.shape[0]):
    lo = np.percentile(arr[b], 2)
    hi = np.percentile(arr[b], 98)
    out[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-8), 0, 1)

# PyGeoVision prepare_for_ai handles all via normalise= parameter:
# "scale_factor" | "minmax" | "percentile" | "none"
```

## 3.8 SAR Preprocessing

SAR requires a specific pipeline order. Despeckle MUST happen in linear power,
before dB conversion:

```python
from pygeovision.data.processors.sar import (
    verify_sar_downloads,
    validate_sar_georeference,
    despeckle_sar,
    linear_to_db,
    normalise_sar_for_ai,
    clip_sar_to_bbox,
)

# Mandatory order: verify -> georeference -> despeckle (LINEAR!) -> dB -> normalise -> clip
check = verify_sar_downloads("./sar/S1_GRD.tif")
assert check["complete"], f"Incomplete: {check['errors']}"

geo   = validate_sar_georeference("./sar/S1_GRD.tif")
src   = geo.repaired_path or "./sar/S1_GRD.tif"

despeckle_sar(src,                  "./sar/S1_desp.tif")
linear_to_db ("./sar/S1_desp.tif", "./sar/S1_db.tif")
normalise_sar_for_ai("./sar/S1_db.tif", "./sar/S1_norm.tif")
clip_sar_to_bbox("./sar/S1_norm.tif", "./sar/S1_clip.tif",
                  bbox_wgs84=(-0.30, 5.50, -0.05, 5.70))
```

## Summary

- `prepare_for_ai()` runs the complete preprocessing pipeline with
  three production bug-fixes embedded.
- SAR has a mandatory processing order: verify, georeference, despeckle
  in LINEAR power, convert to dB, normalise, clip.
- Band stacking requires all inputs at the same resolution.

## Exercises

1. Apply cloud masking to a Sentinel-2 scene. What percentage is cloud-free?
2. Stack 6 bands at 10m. Verify shape is (6, H, W).
3. Compare `scale_factor` vs `percentile` normalisation using `histogram()`.
4. Run `prepare_for_ai()` on a SAR scene — what steps does it report?
