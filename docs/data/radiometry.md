# Radiometry and cloud masks

Satellite files store **digital numbers** (DN), not physical quantities. Using them raw gives indices that are wrong
and thresholds that do not transfer between dates or sensors. PyGeoVision converts every optical scene to **surface
reflectance** before any analysis, masks cloud and shadow, and records where a scene really holds data.

This is done in one shared module, `pygeovision.data.radiometric`, used by all pipelines and available to you.

## Why it matters — three real failure modes it prevents

| Mistake | Effect | How it is avoided |
|---|---|---|
| Treating Sentinel-2 DN/10000 as reflectance after January 2022 | Every band is **0.1 too high**; water looks bright, every normalised-difference index is flattened | The +1000 offset of processing baseline 04.00 is subtracted when the scene id says the product is newer than 2022-01-25 |
| Using one Landsat band map for all missions | On Landsat 8/9 `SR_B2` is *blue*, on Landsat 5/7 `SR_B1` is — the wrong band is silently selected | Band aliases are chosen by mission prefix (`LC08`, `LC09`, `LE07`, `LT05`, `LT04`) |
| Applying the Sentinel-2 formula to anything that "is not Landsat" | NAIP, PlanetScope or MODIS get a quietly wrong reflectance | Sensors are identified positively; an unidentified sensor raises an error unless you pass `apply_scale=False` |

## Scale factors used

| Product | Formula | Source |
|---|---|---|
| Sentinel-2 L2A, baseline < 04.00 | `reflectance = DN / 10000` | ESA product specification |
| Sentinel-2 L2A, baseline ≥ 04.00 (generated on/after 2022-01-25, including reprocessed older archive) | `reflectance = (DN − 1000) / 10000` | ESA; stored as `BOA_ADD_OFFSET` |
| Landsat Collection 2 Level-2 surface reflectance | `reflectance = DN × 2.75e-05 − 0.2` | USGS LSDS-1619 |
| Landsat Collection 2 Level-2 surface temperature | `T[K] = DN × 0.00341802 + 149.0` | USGS LSDS-1619 |

Results are clipped to 0–1. **A pixel with DN 0 in any band is fill** (both programmes use 0) and becomes `NaN` in
every band, never a plausible reflectance of 0.

How the Sentinel-2 baseline is found: from the `_N0510_` token in ESA names; otherwise from the product-generation
timestamp at the end of a Planetary Computer id; otherwise from the acquisition date in an Earth Search id. An old
scene whose id carries none of these is treated as offset-free.

## Cloud and shadow masks

| Sensor | Layer | Masked values |
|---|---|---|
| Sentinel-2 | `SCL` (Scene Classification Layer) | 3 cloud shadow, 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus |
| Landsat | `QA_PIXEL` bit flags | bit 1 dilated cloud, bit 2 cirrus, bit 3 cloud, bit 4 cloud shadow |

Masked pixels are set to `NaN` in every band — **unknown, not clear**. The mask layer is resampled with nearest
neighbour (it is categorical; interpolating class codes would invent classes). Other SCL classes (vegetation,
bare soil, water, snow, saturated pixels) are kept. If the cloud layer was not downloaded, masking is skipped
with a warning — name `SCL` / `QA_PIXEL` in `bands=` when you download yourself.

Masks are probabilistic. Thin haze, small cumulus and the edges of cloud shadows are the usual misses; bright roofs
and sand are the usual false alarms. Cloud *shadow* is a classic source of false water in water maps.

## The footprint layer

For each stacked scene a **footprint raster** is written in a `footprints/` folder beside it: 1 where the scene holds
data at all, regardless of cloud. It lets the pipeline tell *"no imagery here"* (outside the swath) from *"cloud
here"*, which have different remedies (another scene versus another date).

## Using it directly

```python
from pygeovision.data.radiometric import stack_and_prepare_bands

stack = stack_and_prepare_bands(
    download.asset_paths,                 # {"B02": Path, "B03": Path, …, "SCL": Path}
    "work/stack.tif",
    bands=("blue", "green", "red", "nir", "swir1", "swir2"),
    scene_id=download.scene_id,           # sensor and baseline are read from this
)
```

or through the client: `client.data.prepare_stack(asset_paths, "work/stack.tif", bands=(...), scene_id=...)`.

**Band names** you can ask for: `blue`, `green`, `red`, `rededge1` (Sentinel-2 only), `nir`, `swir1`, `swir2`, and
`thermal` (Landsat). They are mapped to the right asset for each sensor (`B02/B03/B04/B05/B08/B11/B12` for
Sentinel-2; `SR_B1…` or `SR_B2…` by Landsat mission).

What you get:

- a GeoTIFF with one band per requested name, in that order, with band descriptions set;
- `float32` reflectance with `NaN` for no-data and cloud (`dtype` stays raw DN if `apply_scale=False`);
- bands at 20 m or 60 m resampled (bilinear) onto the first band's grid so the stack lines up;
- tags `reflectance_scale` and `boa_add_offset_dn` on Sentinel-2 stacks, so the offset used is recorded.

It **raises** instead of guessing when a requested band is missing from the downloaded assets (it lists the assets it
found) or when the sensor cannot be identified and `apply_scale=True`.

| Function | Purpose |
|---|---|
| `identify_sensor(scene_id)` | `(is_landsat, is_sentinel2)`; `(False, False)` means "do not guess" |
| `landsat_mission_aliases(scene_id)` | band-name → asset map for that mission |
| `sentinel2_boa_add_offset(scene_id)` | `1000` or `0` |
| `dn_to_reflectance(dn, scene_id, is_landsat=…)` | the conversion alone, on a `(bands, rows, cols)` array |
| `stack_and_prepare_bands(...)` | the full stacking, scaling and masking |
| `crop_to_bbox(path, bbox)` | crop to a WGS84 box (reprojects the box to the raster CRS; keeps every pixel the box touches; raises if there is no overlap) |
| `real_pixel_area_ha(bounds, width, height, crs)` | true area per pixel — geodesic for degree grids, `res_x × res_y` for metric grids |
| `footprint_path(stack_path)` | where a stack's footprint raster is |

```{warning}
`crop_to_bbox` raises `ValueError` when the box does not intersect the raster. This usually means a search returned a
scene whose *footprint* looked right but whose *data* does not reach the box — try another scene.
```

## Area per pixel in degrees

Pipelines write rasters in degrees (EPSG:4326). Pixel area is not `res × res`: a pixel 0.0001° wide is about 11 m at
the equator and 7 m at 51° N. Reports and statistics use `real_pixel_area_ha` (and per-row ellipsoidal areas), which is
why areas in hectares agree with GIS software. If you compute areas yourself on a degree raster, do the same.
