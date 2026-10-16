# Radar (SAR) preprocessing

Synthetic-aperture radar sees through cloud and at night, which is why flood, change and ship pipelines use it. Its
data behave very differently from optical imagery: values are **backscatter power**, speckle is multiplicative, and the
physically meaningful scale is decibels. This page documents what PyGeoVision does to prepare **Sentinel-1** rasters
(`pygeovision.data.processors.sar`), and — equally important — what it does *not* do.

## What the module does, and in what order

The order is a hard constraint, not a preference.

| Step | Function | What it does | Why this order |
|---|---|---|---|
| S0 | `verify_sar_downloads` / `check_download_complete` | Reads every tile of each downloaded GeoTIFF; reports how many are readable. If a polarisation failed, `fallback_policy="vv_only"` continues with VV | A half-downloaded file passes an "exists" check but fails deep inside processing with an opaque `TIFFFillTile` error |
| S1 | `validate_sar_georeference` | Detects a file tagged with a UTM CRS but holding a **pixel-space identity transform** (pixel size ±1 with no rotation), and recovers the true transform from the embedded ground control points (GCPs); the repaired copy goes to a `_repaired/` folder beside the file | Some reprojection routes write a valid-looking CRS with no real coordinates. Clips then fail with "shapes do not overlap raster" |
| S6 | `despeckle_sar` | Speckle filter on **linear power**: `enhanced_lee`, `refined_lee` or `boxcar`, odd `window_size` (default 7), `num_looks` (default 4) | The filters' statistics assume linear (Gamma-distributed) data. Filtering dB values is mathematically wrong |
| S7 | `linear_to_db` | `10·log10(σ⁰)`, clipped to −35…+5 dB; zero or negative power becomes **NaN**, not −35 dB | dB after despeckle; clipping removes impossible extremes |
| S8 | `normalise_sar_for_ai` | `minmax_db`: fixed bounds −35…+5 dB → 0…1 (default); or `percentile` (2nd/98th, per scene) | Fixed bounds make the scaling scene-independent so a before/after pair is scaled identically |
| S9 | `clip_sar_to_bbox` | Reprojects your WGS84 box into the raster's CRS, then crops; raises `ValueError` if there is no overlap | WGS84 numbers applied to a UTM raster do not overlap it |

`SARPreprocessor(work_dir, bbox_wgs84, filter_type, window_size, num_looks, db_min, db_max).run(raw_path, output_path)`
runs S0 → S1 → S6 → S7 → S8 → S9 and returns the output path (or `None` on failure).

```python
from pygeovision.data.processors.sar import SARPreprocessor

prep = SARPreprocessor(work_dir="./sar_work", bbox_wgs84=(-0.15, 51.47, -0.10, 51.52))
out = prep.run("iw-vv.tiff", "vv_ready.tif", label="2024-06-VV")
```

Result: a single-band `float32` GeoTIFF in 0…1 (NaN where there was no valid backscatter), cropped to your box.

```{warning}
`run()` is forgiving in two places: if despeckling raises, it **continues with the unfiltered data** (a warning is
logged); if the clip fails, it **keeps the full extent**. Check the log, or call the functions yourself when you need
a hard failure.
```

### The speckle filters

| Filter | Method | Cost | Use |
|---|---|---|---|
| `boxcar` | Moving-window mean | lowest | Quick looks; blurs edges |
| `enhanced_lee` | Lee local-statistics filter on the window mean and variance, with noise variance `mean²/num_looks` | low | Default for Sentinel-1 |
| `refined_lee` | Lee weighting using the homogeneous half-window across the dominant edge (four orientations; Lopes et al. 1990) | about 8× `enhanced_lee` | Edge-preserving; slowest |

The name `enhanced_lee` refers to the Lee weighting on window statistics; it does not apply the coefficient-of-variation
class thresholds of the Lopes "enhanced Lee" variant.

## What it does not do

Calibration to σ⁰, thermal-noise removal and terrain correction (range-Doppler / RTC geocoding) are **not performed by
PyGeoVision**. They are done by ESA SNAP or pyroSAR, or are already applied in the product you download. For
Sentinel-1 from Planetary Computer, use the **`sentinel-1-rtc`** collection (radiometrically terrain-corrected
power, ~10 m); the flood pipelines do. GRD products from other sources need these steps outside PyGeoVision first. The
module's input must be **linear power** (not dB).

(The step list `SAR_APPROACH_LIST` in `pygeovision.data.processors.sar_approach_list` documents the full recommended
workflow, including these external steps and the AI steps below, with the failure each step prevents.)

## Using radar with AI models

Optical foundation models (Prithvi, DINOv3) do not accept raw backscatter. The adapters in
`pygeovision.models.adapters` map radar to the input those models expect:
`sar_to_hls_6ch` (physics-guided mapping into six HLS-like channels), `sar_to_pseudo_rgb`, `coregister_sar_pair`
(align a pre/post pair), `validate_sar_ai_input`, `SARPrithviAdapter`, `SARDINOv3Adapter`, and
`SARSpeckleAugmentation` for fine-tuning. See [Foundation models](../foundation-models/radar.md#radar-with-optical-foundation-models).

## Radar pipelines

For flood mapping from radar you do not need these functions — [`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md)
and [`multisensor_flood_mapping`](../pipelines/catalog/multisensor_flood_mapping.md) handle the whole chain.
