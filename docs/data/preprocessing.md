# Preprocessing toolbox

Use this when you have rasters of your own — downloaded scenes, a drone mosaic, a file from a colleague — and want
them **model-ready or analysis-ready**. (Pipelines do their own preparation; see
[Radiometry](radiometry.md) and [Scene selection](scene-selection.md).)

There are two layers:

- `client.preprocess` (`pygeovision.preprocess.Preprocessor`) — PyGeoVision's own rasterio-based operations:
  stack, clip, mask, normalise, resample, mosaic. Each preserves CRS, transform and nodata and validates its output.
- `client.data` / the pygeofetch bridge — pass-throughs to pygeofetch's processing engine for atmospheric
  correction, pan-sharpening, compositing, topographic correction and cloud filling (see
  [below](#operations-delegated-to-pygeofetch)).

## Operations

| Method | Does | Key arguments |
|---|---|---|
| `stack_bands(band_paths, output_path, band_names=None)` | Single-band files → one multi-band GeoTIFF | `band_paths` a list or `{name: path}` |
| `stack_from_dir(scene_dir, band_names, output_path, sensor="auto")` | Finds `B02.tif`… in a Sentinel-2 or Landsat folder and stacks them | `band_names=["B02","B03","B04","B08"]` |
| `clip_to_bbox(path, bbox, output_path=None, bbox_crs="EPSG:4326")` | Crop to a box | `all_touched=` |
| `clip_to_polygon(path, geojson, …)` | Crop and mask to polygons | `crop=True`, `all_touched`, `nodata` |
| `mosaic(paths, output_path=None, method="first")` | Merge overlapping or adjacent rasters | `method` |
| `apply_cloud_mask(path, mask_path, nodata=0.0, invert=False)` | Set pixels where a binary mask (1 = cloud) says so to nodata | |
| `apply_scl(path, scl_path, keep_classes=(4,5,6), nodata=0.0)` | Mask with the Sentinel-2 scene classification | keep classes, see below |
| `normalise(path, method="minmax", scale_factor=10000, percentile=2.0)` | Scale pixel values | `minmax`, `zscore`, `percentile`, `scale_factor` |
| `resample(path, resolution_m, resampling="bilinear")` | New pixel size in metres | `nearest`, `bilinear`, `cubic`, `lanczos`, `average` |
| `pipeline(...)` | All of the above, in the right order, in one call | see below |
| `info(path)` | Summary of a GeoTIFF (size, CRS, bands, resolution…) | |
| `validate(path)` | Run the [validator](validation.md) | |

### Normalisation methods

| Method | Result | Use for |
|---|---|---|
| `scale_factor` | `x / scale_factor` (10 000 for Sentinel-2 L2A) | Raw DN → 0–1 reflectance. **Check the offset note in [Radiometry](radiometry.md)** — for products after January 2022 you must also subtract 1000 first, which this method does not do; use `stack_and_prepare_bands` for scenes |
| `minmax` | each band to 0–1 by its own min and max | Quick scaling; sensitive to outliers; **per-image**, so two scenes are not comparable |
| `percentile` | clip to the `[p, 100−p]` percentiles, then 0–1 | Robust stretch for display and models |
| `zscore` | `(x − mean)/std` per band | Models that expect standardised inputs |

### Sentinel-2 scene classification (SCL) classes

`1` saturated, `2` dark area, `3` cloud shadow, `4` vegetation, `5` bare soil, `6` water, `7` unclassified,
`8` cloud medium, `9` cloud high, `10` thin cirrus, `11` snow. The default `keep_classes=(4, 5, 6)` **removes dark
areas, unclassified pixels and snow** as well as cloud. Use `(2, 4, 5, 6, 7, 11)` to remove only cloud, shadow, cirrus
and saturation. (The pipelines mask 3, 8, 9 and 10 only.)

## One call: `pipeline()`

```python
pre = client.preprocess
out = pre.pipeline(
    input_path="./downloads/S2C_20240628/",          # a scene folder with B02.tif, B03.tif…
    output_path="ready.tif",
    stack_bands=["B02", "B03", "B04", "B08", "B11", "B12"],
    bbox=(-74.1, 40.6, -73.7, 40.9),
    scl_path="./downloads/S2C_20240628/SCL.tif",
    normalise="scale_factor",
)
print(out["steps_applied"], out["shape"], out["resolution_m"])
```

Order: **stack → clip → cloud mask (binary mask, then SCL) → normalise → resample**. Each step is skipped when its
argument is `None`. Returns `{"output_path", "steps_applied", "shape", "resolution_m"}`. Intermediate files are
deleted unless `keep_intermediates=True`. `clip_geojson` takes precedence over `bbox`.

Masked pixels are written as `nodata` (default `0.0`). That is the right convention for model input, but a `0` that
means "masked" looks like a real value to anything that does not read the nodata tag — check `info(out)`.

### Command line

```bash
pygeovision preprocess stack ./S2C_20240628/ --bands B02,B03,B04,B08,B11,B12 --output stack.tif --validate
pygeovision preprocess clip stack.tif --bbox -74.1,40.6,-73.7,40.9
pygeovision preprocess normalise stack.tif --method scale_factor --scale-factor 10000
pygeovision preprocess resample s2_20m.tif --resolution 10 --method bilinear
pygeovision preprocess pipeline ./S2C_20240628/ --output ready.tif \
    --bands B02,B03,B04,B08,B11,B12 --bbox -74.1,40.6,-73.7,40.9 --scl SCL.tif --normalise scale_factor
```

Note the CLI `--bbox` here is **comma-separated** (`minlon,minlat,maxlon,maxlat`), unlike `channel` and `data search`,
which take four space-separated numbers. `--validate` re-checks the output. Options per command:
[CLI reference](../reference/cli/index.md).

## Operations delegated to pygeofetch

When pygeofetch's processing engine is installed (the `doctor` command reports it), these methods of the bridge are
available through `client._pgf_bridge`; each returns an output path:

| Method | Purpose |
|---|---|
| `preprocess_atmos(path, method="dos1")` | Atmospheric correction |
| `preprocess_cloud_mask(path, method="scl")` | Cloud masking |
| `preprocess_clip`, `preprocess_reproject`, `preprocess_resample` | Geometry |
| `preprocess_pansharpen(pan, ms, method="brovey")` | Pan-sharpening (Brovey or IHS) |
| `preprocess_tile(path, tile_size=512, overlap=64)` | Cut a scene into training chips |
| `preprocess_mosaic`, `preprocess_composite(inputs, method="median")` | Mosaic; temporal composite (median/mean/max/best-pixel) |
| `preprocess_topo_correct(path, dem, method="c_correction")` | Illumination correction in mountains with a DEM |
| `preprocess_cloud_fill(cloudy, time_series)` | Fill cloud gaps from other dates |
| `batch_process(inputs, chain, output_dir, parallel=4)` | Run a chain on many scenes |

The same chain is available as `client.pgf_pipeline("name")`.

Without the processing engine, the operations that have a PyGeoVision equivalent — clip, reproject, resample, mosaic,
composite, cloud mask — use the built-in version (the results are documented above). The five that have **no**
PyGeoVision implementation — atmospheric correction, pan-sharpening, tiling into chips, topographic correction and cloud
filling — raise `PreprocessingError` rather than return the input unchanged or the output of a different operation.
(For Sentinel-2 L2A and Landsat Level-2 data, atmospheric correction has already been done by the provider; what you
need is the [scaling in Radiometry](radiometry.md). For chips, see [Training](../training/index.md).)

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `Could not find band(s) […]` from `stack_from_dir` (it lists the files it saw) | File names do not follow `…_B04.tif` / `…_B04_10m.tif`, or the band was not downloaded | Download the missing band, or pass explicit paths to `stack_bands` |
| Output is all zeros | The mask removed everything (`keep_classes` too narrow) or the bbox is outside the raster | `info(path)`; widen `keep_classes`; check `bbox` and `bbox_crs` |
| Two scenes look different after `minmax` | Per-image min/max | Use `scale_factor` (reflectance) or one shared stretch |
| Mosaic shows seams | Scenes from different dates or calibrations | Correct each scene first, then mosaic; or `preprocess_composite` |
