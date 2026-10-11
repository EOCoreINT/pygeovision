# Validation

`DataValidator` checks a raster or array **before a model sees it**. A model never complains about a bad input: it
runs, and returns a confident wrong answer. Validation turns silent problems (NaN, wrong range, wrong band count,
mismatched CRS) into reports.

```python
client = pgv.PyGeoVision()
report = client.validator.validate("scene.tif", required_bands=6, value_range=(0, 1), target_crs="EPSG:4326")
print(report.summary())
report.raise_if_failed()
```

## Modes

| `DataValidator(mode=…)` | On errors |
|---|---|
| `"fix"` (default; what `client.validator` uses) | Fix what can be fixed **in memory**, log a warning, return the report |
| `"strict"` | Raise `ValidationError` |
| `"warn"` | Log, never raise |

Also: `max_nodata_pct` (default 50), `outlier_sigma` (default 4.0).

**Fixes never modify your file.** They are applied to an in-memory copy; the file on disk is untouched. To keep a fixed
version, use `validate_for_inference` (returns the array) and write it yourself.

## Checks

| Check | Looks for | Auto-fix (in memory) |
|---|---|---|
| `check_nulls` | NaN and Inf | Replaced by **0** |
| `check_dtype` | dtype ≠ `expected_dtype` | Cast |
| `check_shape` | shape ≠ `expected_shape` | none (error) |
| `check_bands` | fewer bands than `required_bands` | none (error) |
| `check_bounds` | values outside `value_range` | Clipped |
| `check_nodata` | more than `max_nodata_pct` of pixels equal to the file's nodata (0 if none is set) | none (warning: use a clearer scene) |
| `check_outliers` | values beyond `outlier_sigma` standard deviations of the band mean | Clipped |
| `check_crs` | no CRS (error), or CRS ≠ `target_crs` (warning) | none |

`ValidationReport`: `source`, `passed` (no *errors*), `issues` (each with `severity`, `check`, `message`, `fix`),
`stats` (bands, size, dtype, min/max/mean/std, `nan_pct`, `zero_pct`), `auto_fixed`; `.errors`, `.warnings`,
`.summary()`, `.raise_if_failed()`.

## Preparing input for a model

```python
arr = client.validator.validate_for_inference("scene.tif", model_type="segmentation")   # (C, H, W) float32
```

| `model_type` | Minimum bands | Range enforced |
|---|---|---|
| `segmentation`, `classification` | 1 | 0…1 |
| `detection` | 3 | 0…1 |
| `change_detection` | 4 | 0…1 |
| `foundation` | 3 | −3…3 (standardised) |
| `regression` | 1 | none |

`validate_for_inference` casts to `float32`, **replaces NaN/Inf with 0**, clips outliers, **repeats bands cyclically** if
there are fewer than the model type needs, and — if values fall outside the range and `auto_normalise=True` — rescales
each band by its own min and max.

```{warning}
Three of those are convenient but lossy: cloud-masked NaN becomes 0 (a real-looking value), repeated bands add no
information, and per-band min/max scaling depends on the image, so two scenes are scaled differently. For
reproducible models, scale with a **fixed** rule (reflectance 0–1 from
[Radiometry](radiometry.md); fixed dB bounds for radar) and use this call as a safety net, not as the scaler.
```

`validate_for_training(images, labels=None, num_classes=None)` checks a set of chips: consistent shapes, and that no label
value reaches `num_classes`. It samples the first five files of a directory.

## Reports

```python
client.validator.generate_report("scene.tif", "validation.html")            # or fmt="json"
```

## Georeference validation

`pygeovision.data.validators.georeference.validate_georeference(path, expected_crs=None, …)` (and the radar variant
`validate_sar_georeference`) check that a raster's transform holds real coordinates: pixel size within
0.5 m–10 km for projected data, ≥ 1e-7° for geographic. A file whose transform is the pixel-space identity (pixel
size ±1, no rotation) is reported as corrupt; if it carries ground control points (GCPs), the true transform is derived
from them and a repaired copy is written to a `_repaired/` folder beside it (`repaired_path`). `reproject_bbox_to_raster_crs(bbox, raster)`
converts a WGS84 box to a raster's CRS. `check_download_complete(path)` reads the tiles and reports `{"complete",
"readable_tiles", "total_tiles"}`.

## Command line

```bash
pygeovision validate run scene.tif --required-bands 6 --value-range 0,1 --target-crs EPSG:4326
pygeovision validate run scene.tif --report validation.html --report-fmt html --mode strict
pygeovision validate for-inference scene.tif --model-type segmentation
```

`--mode strict|fix|warn` (default `fix`). In strict mode a failed validation raises an error and the command exits non-zero. `for-inference` prints the shape, dtype and range of the prepared array (`✓ Ready for segmentation | shape=(6, 512, 512) dtype=float32 range=[0.0000, 1.0000]`); it does not write a file.

## What to do when…

| Report says | Meaning | Do this |
|---|---|---|
| `NaN value(s) found` | Cloud-masked or fill pixels | Expected for pipeline outputs. Mask rather than zero-fill if no-data must be preserved |
| `% of pixels are no-data` | Mostly empty | Another scene or date; check the footprint |
| `CRS mismatch` | Different projection than expected | Reproject first; do not feed mixed-CRS pairs to change detection |
| `Too few bands` | Wrong stack | Re-stack with the expected bands in the expected order |
| `outlier(s) beyond 4σ` | Bright cloud edges, saturated roofs, or genuine extremes | Inspect; clipping may remove real signal |
