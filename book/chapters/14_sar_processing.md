# Chapter 14: SAR Processing

## 14.1 Understanding SAR Data

Synthetic Aperture Radar (SAR) transmits microwave pulses and measures backscatter.
Unlike optical sensors, SAR operates independently of sunlight and clouds.

| Property | Optical | SAR |
|----------|---------|-----|
| Solar illumination | Required | Not needed |
| Cloud penetration | No | Yes |
| Phase info | No | Yes (SLC) |
| Flood detection | Difficult in clouds | Always available |

## 14.2 The Three Production Bug-Fixes

### BUG 1: Identity-transform CRS corruption
```python
from pygeovision.data.validators.georeference import validate_georeference

result = validate_georeference("./sar/S1_reprojected.tif")
working = result.get("repaired_path") or "./sar/S1_reprojected.tif"
```

### BUG 2: Partial download detection
```python
from pygeovision.data.validators.georeference import check_download_complete

check = check_download_complete("./sar/S1_GRD.tif")
if not check["complete"]:
    raise ValueError(f"Incomplete download: {check['errors']}")
```

### BUG 3: WGS84 vs UTM CRS mismatch
```python
from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs

bbox_utm = reproject_bbox_to_raster_crs((-0.30, 5.50, -0.05, 5.70), "./sar/S1.tif")
```

## 14.3 Complete SAR Pipeline

```python
from pygeovision.data.processors.sar import (
    verify_sar_downloads, validate_sar_georeference,
    despeckle_sar, linear_to_db, normalise_sar_for_ai, clip_sar_to_bbox,
)

# Correct order: verify -> georeference -> despeckle(LINEAR!) -> dB -> normalise -> clip
check = verify_sar_downloads("./sar/S1_GRD.tif")
assert check["complete"]

geo     = validate_sar_georeference("./sar/S1_GRD.tif")
working = geo.repaired_path if not geo.valid else "./sar/S1_GRD.tif"

despeckle_sar(working,               "./sar/S1_desp.tif", filter_type="enhanced_lee")
linear_to_db ("./sar/S1_desp.tif",  "./sar/S1_db.tif")
normalise_sar_for_ai("./sar/S1_db.tif", "./sar/S1_norm.tif")
clip_sar_to_bbox("./sar/S1_norm.tif", "./sar/S1_clip.tif",
                  bbox_wgs84=(-0.30, 5.50, -0.05, 5.70))
```

**Critical:** Despeckle MUST happen before dB conversion. Speckle follows
a Gamma distribution in linear power — not in dB space.

## 14.4 SAR Flood Detection

```python
from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
import rasterio, numpy as np

with rasterio.open("./sar/S1_clip.tif") as src:
    vv = src.read(1).astype("float32")[np.newaxis]
    vh = src.read(2).astype("float32")[np.newaxis]

adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
result  = adapter.run(vv, vh)
mask    = result["prediction"]
print(f"Flooded: {mask.sum()*100/1e6:.1f} km2 at 10m resolution")
```

## 14.5 SAR Change Detection (Damage Assessment)

```python
from pygeovision.models.change_detection.changeformer import ChangeDetection

cd = ChangeDetection(model_variant="changeformer", in_channels=2, num_classes=4)
cd.build()
damage = cd.detect("./sar/S1_pre.tif", "./sar/S1_post.tif",
                    output_path="./sar/damage.tif")

for i, cls in enumerate(["Stable","Minor","Moderate","Severe"]):
    pct = (damage.prediction == i).mean() * 100
    print(f"  {cls}: {pct:.1f}%")
```

## 14.6 Oil Spill Detection

Oil slicks suppress SAR backscatter (specular reflection). The same dark-pixel
algorithm that detects floods also detects oil spills on the sea surface.

```python
with rasterio.open("./sar/S1_maritime.tif") as src:
    vv = src.read(1).astype("float32")

sea_bg    = float(np.nanmedian(vv))
oil_mask  = (vv < sea_bg * 0.4).astype("uint8")
oil_area  = oil_mask.sum() * 100 / 1e6
print(f"Oil slick: {oil_area:.2f} km2")
```

## Summary

- SAR is cloud-independent and works at night — critical for disasters.
- Three production bug-fixes: CRS validation, download completeness, CRS-aware clip.
- Mandatory order: verify -> georeference -> despeckle (LINEAR!) -> dB -> normalise -> clip.

## Exercises

1. Download S1 GRD and run the complete pipeline.
2. Detect a flood from a real event.
3. Compare Enhanced Lee vs Boxcar despeckle filters visually.
