# Canopy height model

**What it computes.** The surface model (DSM) is the highest point in each cell. The terrain model (DTM) is the mean height of
the points classified as ground (class 2). CHM = DSM − DTM, never negative.

```python
r = proc.canopy_height_model("tile.las", "chm.tif", resolution=1.0, filter_ground=True, exclude_noise=True)
```

| Argument | Default | Meaning |
|---|---|---|
| `resolution` | 1.0 | Cell size in the file's units (metres for projected data). The raster covers the points' bounding box in cells of **exactly** this size |
| `filter_ground` | `True` | DTM from class-2 points. `False`: the lowest point per cell is taken as ground (use for unclassified files) |
| `exclude_noise` | `True` | Drop points in LAS noise classes 7 and 18 before modelling |

How the DTM gaps are filled: a cell that has points but no ground point takes the DTM of the **nearest** cell that has one.
That is fine on flat or gently sloping terrain and blocky on steep slopes.

## Expected output

```text
{
 "success": true, "n_points": 13900, "resolution": 1.0, "crs": "EPSG:32630",
 "grid": {"width": 100, "height": 100},
 "empty_cells_fraction": 0.324,
 "height_stats": {"max_m": 19.4, "mean_m": 3.4, "coverage": 0.521, "canopy_cover_2m": 0.177}
}
raster: 100 x 100 cell (1.0, 1.0) crs EPSG:32630 nodata -9999.0 max 19.4
```

* The GeoTIFF is float32, one band, LZW, **nodata = −9999**, in the file's CRS. A file without a CRS gives a GeoTIFF without
  one and a logged warning; set it yourself.
* **`empty_cells_fraction`** is the share of cells that received no point at all (here 32 %, because the demo has only about
  one ground return per m²). They are nodata, not 0 m: "no data" and "no canopy" are different. If this number is large,
  use a coarser `resolution`; as a guide you want several points per cell.
* `n_points` is 13 900, not 13 902: the two noise points were excluded. Without that exclusion (the demo also shows
  `exclude_noise=False`) the maximum height becomes **298 m**, from two stray returns.
* The true tallest tree is below 20 m and the model reports 19.4 m. **`max_m` is a single cell**: sensitive to one erroneous
  point. For forest reporting use a high percentile of the raster, not the maximum.
* **`coverage`** is the share of data cells with CHM > 0, which includes tiny positive values from ground noise (0.05 m).
  **`canopy_cover_2m`** (share ≥ 2 m, the usual vegetation threshold) is the number to quote.

## No ground classification

```text
ValueError: .../unclassified.las has no ground points (LAS class 2), so no terrain model can be built. Classify the gro ...
filter_ground=False -> max_m 18.7 canopy_cover_2m 0.119
```

With `filter_ground=True` and no class-2 points the function **raises** (an earlier version warned "CHM will be all zeros" and
wrote absolute elevations instead). With `filter_ground=False` the lowest return in each cell stands in for the ground. The
estimate is lower than the truth under dense canopy (cover 0.119 versus 0.177 with real ground), because in a cell fully
covered by crowns the lowest return may be a branch. Treat it as a minimum.
