# Appendix D: Troubleshooting

## Data Acquisition Issues

### "shapes do not overlap raster"

**Cause:** Clipping WGS84 bbox to a UTM raster — BUG 3.

**Fix:**
```python
from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs

bbox_utm = reproject_bbox_to_raster_crs(bbox_wgs84, raster_path)
# Or use clip_sar_to_bbox() which handles this automatically
```

### "Identity-transform CRS detected"

**Cause:** Reprojection produced pixel_width=1.0, origin=(0,0) — BUG 1.

**Fix:**
```python
from pygeovision.data.validators.georeference import validate_georeference

result  = validate_georeference("./scene.tif")
working = result.get("repaired_path") or "./scene.tif"
```

### "Incomplete download" / file is too small

**Cause:** Chunked download interrupted — BUG 2.

**Fix:**
```python
from pygeovision.data.validators.georeference import check_download_complete

check = check_download_complete("./sar/S1_GRD.tif")
if not check["complete"]:
    # Re-download the file
    client.download(results, output_dir="./sar/", resume=False)
```

## SAR Issues

### Output looks wrong after despeckle

**Cause:** Despeckle was applied after dB conversion.

**Fix:** Always despeckle in LINEAR power:
```python
despeckle_sar(input_path, desp_path, filter_type="enhanced_lee")
linear_to_db(desp_path, db_path)   # THEN convert to dB
```

### No flood pixels detected

**Cause:** VH threshold too high, or image is dry season.

**Fix:**
```python
# Check VH statistics
import numpy as np, rasterio
with rasterio.open("./sar/S1_norm.tif") as src:
    vh = src.read(1)
print(f"VH range: {vh.min():.3f} to {vh.max():.3f}")
print(f"VH mean: {vh.mean():.3f}")
# VH < 0.15 is typically water/flood for normalised [0,1] data
```

## AI Inference Issues

### "Input validation failed" from Prithvi

**Cause:** Wrong band count or normalisation range.

**Fix:**
```python
from pygeovision.models.foundation.prithvi import validate_prithvi_input

val = validate_prithvi_input(arr, source="sentinel2", n_prithvi_bands=6)
print(val)   # shows which check failed
# arr must be (6, H, W), float32, values in [0, 1]
```

### mIoU is very low (< 0.3)

**Causes and fixes:**
1. Wrong band order — check HLS order: Blue, Green, Red, NIR, SWIR1, SWIR2
2. Wrong normalisation — use scale_factor=10000 for Sentinel-2 L2A
3. Label mismatch — verify class IDs match model training

## Performance Issues

### Out of memory during inference

**Fix:** Use tiled inference:
```python
from pygeovision.inference.tiled import TiledInference

tiler = TiledInference(model=model, tile_size=256, overlap=32)
pred  = tiler.predict("./data/large_scene.tif", output_path="./pred.tif")
```

### Slow downloads

**Fix:** Increase workers:
```python
downloads = client.download(results, max_workers=8, skip_existing=True)
```

## Common Errors Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `shapes do not overlap raster` | WGS84 bbox on UTM raster | BUG 3: use `clip_sar_to_bbox()` |
| `identity-transform CRS` | Corrupted georeference | BUG 1: `validate_sar_georeference()` |
| `source shape inconsistent` | Wrong numpy array shape | Use 2D array for single band write |
| `No module named torch` | PyTorch not installed | `pip install "pygeovision[train]"` |
| `API key not found` | No ANTHROPIC_API_KEY | Set env var or use heuristic planner |
