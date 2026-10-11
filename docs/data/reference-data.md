# Reference products and terrain

Some questions are best answered by a **published product** rather than by processing imagery yourself: a land-cover
map, a water-history atlas, a forest/non-forest map, a DEM. PyGeoVision reads only the **window of the product that
your box needs** from Cloud-Optimized GeoTIFFs, puts it on a metric grid, and measures it. It never classifies,
thresholds or calibrates these products; what you get is the product, clipped, measured and (for change pipelines)
differenced.

## Why this approach

- The tiles are huge (a single COG can cover thousands of kilometres). Downloading whole files is not an option; windowed
  reads fetch a few megabytes.
- A single, **metric** grid (the local UTM zone) gives every pixel the same known area, so hectare figures are exact.
- Where no tile has data, pixels stay **invalid** — they are never filled with a plausible value.

## Products used

| Pipeline | Product | Resolution | Years | Source collection |
|---|---|---|---|---|
| [`land_cover`](../pipelines/catalog/land_cover.md) | ESA WorldCover 2021 (or Dynamic World) | 10 m | 2021 | ESA WorldCover public bucket |
| [`landcover_change`](../pipelines/catalog/landcover_change.md), [`urban_expansion`](../pipelines/catalog/urban_expansion.md) | Impact Observatory / Esri annual land cover (`io-lulc-annual-v02`) | 10 m | 2017–2023 | Planetary Computer |
| [`forest_change`](../pipelines/catalog/forest_change.md) | JAXA PALSAR forest / non-forest | 25 m | 2015–2020 | Planetary Computer |
| [`surface_water_dynamics`](../pipelines/catalog/surface_water_dynamics.md) | JRC Global Surface Water history | 30 m | 1984–2020 | Planetary Computer |
| [`evapotranspiration`](../pipelines/catalog/evapotranspiration.md) | MODIS MOD16A3GF annual ET and PET | 500 m | annual | Planetary Computer |
| [`active_fire`](../pipelines/catalog/active_fire.md) | MODIS MOD14A1 daily thermal anomalies | 1 km | daily | Planetary Computer |
| [`biodiversity_hotspot`](../pipelines/catalog/biodiversity_hotspot.md), [`wind_farm_siting`](../pipelines/catalog/wind_farm_siting.md) | ESA WorldCover 2021 | 10 m | 2021 | WorldCover public bucket |
| terrain pipelines | Copernicus GLO-30 DEM | 30 m | – | Planetary Computer, AWS Open Data, or OpenTopography |

Every product's own caveats are copied into `stats["product_note"]` and into the report. Read them: a map product's
"change" between years includes classifier disagreement, not only real change.

## Digital elevation models

`terrain_analysis`, `drainage_analysis`, `solar_potential`, `archaeological_site`, `wind_farm_siting` and `multisensor_flood_mapping` (HAND) read a DEM.

| `dem_source` | Product | Account | Notes |
|---|---|---|---|
| `copernicus` (default) | Copernicus GLO-30, from Planetary Computer | none | Tried first |
| `copernicus_aws` | The same product, from AWS Open Data | none | Also the **automatic fallback** if the Planetary Computer read fails (for example HTTP 403) |
| `opentopography` | SRTM, Copernicus 30/90, ALOS… (`dem_type=`) | API key | `pygeovision data auth add opentopography --api-key …` |

```{important}
The Copernicus DEM is a **surface** model: it includes tree canopy and buildings. Heights are raised there, and slope
and drainage are distorted in forest and towns. It is not a bare-earth terrain model.
```

## The reading API

For your own analysis, the module `pygeovision.data.reference` exposes the pieces:

```python
from pygeovision.data.reference import PlanetaryComputer, make_grid, read_to_grid

bbox = (-1.62, 6.60, -1.55, 6.66)
grid = make_grid(bbox, res=30)                       # 30 m pixels in the local UTM zone (snapped to multiples of 30)
pc = PlanetaryComputer()
items = pc.search("cop-dem-glo-30", bbox)            # STAC /search, all pages
urls = pc.asset_urls("cop-dem-glo-30", items, "data")  # signed URLs
dem = read_to_grid(urls, grid, resampling="bilinear")  # window-read + mosaic (first valid pixel wins)
print(dem.valid.mean(), grid.pixel_area_ha)
```

| Function | Purpose |
|---|---|
| `make_grid(bbox, res, crs=None, max_pixels=250_000_000)` | North-up metric grid covering the box; UTM zone chosen automatically; refuses grids over `max_pixels` |
| `read_to_grid(sources, grid, …)` | Reads each source's overlapping window, warps and mosaics; `invalid_values=` marks product fill codes; `skip_missing=True` tolerates a missing tile |
| `PlanetaryComputer.search(collection, bbox, datetime=None)` | STAC `/search` (the correct endpoint; `/collections/<id>/items` can return the wrong scenes for a date window), all pages |
| `PlanetaryComputer.sign(href, collection)` | Signed URL for blob storage |
| `copernicus_dem_aws_urls(bbox)` | The 1° GLO-30 tiles covering a box on AWS |
| `utm_crs_for(lon, lat)` | UTM EPSG code of a point |
| `parse_year(value, what)` | `2021`, `"2021"`, `"2021-06"` → `2021` |

A source may be a URL or a **local file path**, which is how the offline tests exercise the same code on small files.
Failures raise `ReferenceDataError` with the message saying what to change (usually the box, the year range, or
network access).

## Choosing years

Annual products accept a year as `2021`, `"2021"` or a date such as `"2021-06"` (the year is used). Each pipeline page
states its valid range. Asking for a year outside the range is an error, not a nearest-year substitution.

## Grid size and speed

`resolution_m=None` lets each pipeline pick a pixel size so that the grid has at most ~60 million cells
(`AUTO_MAX_PIXELS`); pass `resolution_m=` to force a coarser grid (faster, less detail). For hydrology, the grid is also
limited by `max_cells` (default 6 million) because flow routing costs about a minute per five million cells.

## What to do when…

| You see | Do this |
|---|---|
| `ReferenceDataError: no tiles` | The box is outside the product's coverage (for example forest/non-forest outside 2015–2020, or ocean) — check the year range and the box |
| HTTP 403 reading the DEM | Automatic fallback to AWS happens; if both fail, `dem_source="opentopography"` with a key |
| Grid refused as too large | Smaller box or larger `resolution_m` |
| Figures differ from a GIS clip of the same product | Compare on the same grid; PyGeoVision snaps to whole multiples of the resolution in UTM and uses nearest neighbour for classes |
