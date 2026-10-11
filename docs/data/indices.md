# Spectral indices

A spectral index combines two or more bands into one number that highlights a surface property — green vegetation,
open water, burned ground, snow. PyGeoVision provides **21** (19 indices and two transforms), each validated before and
after computation, usable on a GeoTIFF or on arrays.

```python
client = pgv.PyGeoVision()
ndvi_path = client.indices.ndvi("stack.tif", red_band=3, nir_band=4, output_path="ndvi.tif")
```

## Why use these instead of writing the formula yourself

- **One tested definition.** Formulas, literature sources and the clipping range are fixed and documented below.
- **Nodata handled.** File-to-file computation writes a nodata value (`-9999`) rather than a fake zero (see
  [Nodata behaviour](#nodata-behaviour)).
- **Division-by-zero safe**, and values clipped to the index's physical range.

## Band numbering — read this first

All index functions take **1-based band positions in your file** (`red_band=3` means "the third band"). The defaults
assume the six-band stack the pipelines and `pygeovision preprocess stack` produce:

| Position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Band | blue (B02) | green (B03) | red (B04) | NIR (B08) | SWIR1 (B11) | SWIR2 (B12) |

If your file is ordered differently, **pass the positions explicitly**. A wrong position does not fail — it computes a
different, valid-looking number.

```{warning}
**NDRE needs a red-edge band** (Sentinel-2 B05), which the default six-band stack does not contain. `ndre()` takes
`red_edge_band` (default 5 only suits a stack ordered B02, B03, B04, B08, B05…). `indices compute … ndre` requires
`--red-edge N`, and `indices all` / `compute_all` skip NDRE unless you give `band_map={"red_edge": N}`.
```

## The catalogue

All are in `pygeovision.data.indices.SpectralIndices` (`client.indices`).

### Vegetation

| Index | Formula | Range kept | Use it for | Beware |
|---|---|---|---|---|
| **NDVI** | (NIR − Red) / (NIR + Red) | −1…1 | Green vegetation amount and vigour. Healthy canopy 0.6–0.9; bare soil 0.1–0.2; water < 0 | Saturates over dense canopy; soil and atmosphere affect it |
| **EVI** | G (NIR − Red) / (NIR + C1 Red − C2 Blue + L), G = 2.5, C1 = 6, C2 = 7.5, L = 1 | −1…1 | Dense canopy where NDVI saturates; reduces atmospheric and soil effects (Huete 2002) | Needs a blue band that is well corrected |
| **SAVI** | (NIR − Red)(1 + L) / (NIR + Red + L), L = 0.5 | −1.5…1.5 | Sparse vegetation over bright soil (Huete 1988) | Choice of L matters; 0.5 is intermediate cover |
| **MSAVI** | (2 NIR + 1 − √((2 NIR + 1)² − 8 (NIR − Red))) / 2 | unclipped | Same, without choosing L (Qi 1994) | |
| **ARVI** | (NIR − rb) / (NIR + rb), rb = Red − γ (Blue − Red) | −1…1 | Hazy atmosphere (Kaufman & Tanré 1992) | Needs blue; γ = 1 default |
| **NDRE** | (NIR − RedEdge) / (NIR + RedEdge) | −1…1 | Chlorophyll in dense canopies (Gitelson & Merzlyak 1994) | **Requires a red-edge band** |
| **RVI** | NIR / Red | 0…30 | Biomass ratio (Jordan 1969) | Unbounded; sensitive to dark red |
| **WDRVI** | (α NIR − Red) / (α NIR + Red), α = 0.1 | −1…1 | Dense, high-biomass canopies (Gitelson 2004) | Values are not comparable to NDVI |
| **VARI** | (Green − Red) / (Green + Red − Blue) | −1…1 | Vegetation from RGB only (Gitelson 2002) | Visible light only: low precision |
| **ExG** | 2g − r − b on normalised RGB | unclipped | Vegetation in RGB / drone images (Woebbecke 1995) | Needs RGB, not reflectance |

### Water, moisture

| Index | Formula | Use it for | Beware |
|---|---|---|---|
| **NDWI** | (Green − NIR) / (Green + NIR) | Open water (McFeeters 1996). Positive over water | Built-up land can also be positive |
| **MNDWI** | (Green − SWIR1) / (Green + SWIR1) | Open water in built-up areas; better than NDWI there (Xu 2006) | Needs SWIR; 20 m native |
| **LSWI** | (NIR − SWIR1) / (NIR + SWIR1) | Canopy and soil water content (Xiao 2004) | Not open-water detection |
| **WRI** | (Green + Red) / (NIR + SWIR1) | Water ratio (Shen & Li 2010) | Range clipped to 0…10 |

### Built-up, soil, fire, snow

| Index | Formula | Use it for | Beware |
|---|---|---|---|
| **NDBI** | (SWIR1 − NIR) / (SWIR1 + NIR) | Built-up surfaces (Zha 2003) | Bare soil also rises |
| **BSI** | ((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)) | Bare soil (Rikimaru 2002) | Dry fields and sand |
| **NBR** | (NIR − SWIR2) / (NIR + SWIR2) | Burned area, with its difference dNBR (Key & Benson) | Use before *and* after: [wildfire_severity](../pipelines/catalog/wildfire_severity.md) |
| **BAI** | 1 / ((0.1 − Red)² + (0.06 − NIR)²) | Burn scars (Martín 1998) | Clipped 0…5000 |
| **NDSI** | (Green − SWIR1) / (Green + SWIR1) | Snow and ice (Hall 1995); > 0.4 is snow | Cloud also bright in green |

### Transforms

| Name | What it produces | Notes |
|---|---|---|
| **TCT** (`tct`) | Tasseled-cap *brightness*, *greenness*, *wetness* from the six bands | `sensor="sentinel2"` (Nedkov 2017 coefficients) or `"landsat8"`. Needs all six bands in blue-to-SWIR2 order. Returns a dict; with `output_prefix` writes `…_tct_B.tif`, `…_tct_G.tif`, `…_tct_W.tif` |
| **PCA** (`pca`) | Principal components of the bands by SVD | `n_components` (default 3); writes `…_pc1.tif`… with `output_prefix` |

## Nodata behaviour

| Input | Output | Pixels with no data (NaN) |
|---|---|---|
| A file, with `output_path=` | `float32` GeoTIFF, nodata `-9999` | stay nodata |
| A NumPy array, or a file without `output_path` | array | **become 0** |

When the result is returned as an array, NaN and Inf are replaced by 0, which is a valid index value. **Apply your
cloud mask afterwards**, or write to a file. The pipelines compute their own indices with `NaN` kept and never use this
array fallback.

Values are also clipped to the "Range kept" column, so a hard ceiling in a plot is a clip, not a property of the scene.

## Compute many at once

```python
paths = client.indices.compute_all("stack.tif", indices=["ndvi", "ndwi", "nbr"], output_dir="./indices")
# {'ndvi': './indices/ndvi.tif', 'ndwi': './indices/ndwi.tif', 'nbr': './indices/nbr.tif'}
```

`compute_all` supports: ndvi, evi, savi, msavi, arvi, ndre*, rvi, wdrvi, ndwi, mndwi, lswi, ndbi, bsi, nbr, bai, ndsi
(*only with a `red_edge` band). `vari`, `exg`, `wri`, `tct` and `pca` are called individually. Unknown names are
skipped with a warning. Pass `band_map={"red": 3, "nir": 4, …}` for other layouts.

## Command line

```bash
pygeovision indices list                                 # the 21 indices, name and category
pygeovision indices compute stack.tif ndvi --output ndvi.tif
pygeovision indices compute stack.tif nbr --nir 4 --swir2 6
pygeovision indices compute stack.tif ndre --red-edge 5 --nir 4     # red-edge band required
pygeovision indices all stack.tif --output-dir ./indices --indices ndvi,evi,nbr,ndwi
```

`compute` options: `--output`, `--red`, `--nir`, `--green`, `--blue`, `--swir1`, `--swir2` (1-based positions),
`--red-edge`, `--validate` (re-validates the written file). `compute` handles ndvi, evi, savi, msavi, ndwi, mndwi,
ndbi, bsi, nbr, bai, ndsi, rvi, arvi, lswi, ndre, wdrvi, vari, exg; use Python for `wri`, `tct` and `pca`.

Output: `✓ NDVI → ndvi.tif`.

## Choosing an index

```{list-table}
:header-rows: 1
:widths: 35 65
* - You want to…
  - Start with
* - Measure greenness
  - NDVI; EVI if the canopy is dense; SAVI if cover is sparse
* - Map open water
  - MNDWI (built-up areas) or NDWI; see the [water pipelines](../pipelines/index.md)
* - Map burned area
  - NBR before and after → dNBR
* - Map snow
  - NDSI > 0.4
* - Map built-up land
  - NDBI, but prefer a land-cover product for counting
```

## Combining with pipelines

Pipelines that need an index (`crop_health`, `flood_mapping`, `water_bodies`, `snow_cover`, `wildfire_severity`…)
compute it themselves, from reflectance with cloud kept as `NaN`, and document the formula on their page. Use this
module for your own analysis on your own rasters.
