# Postprocessing toolbox

A model or a pipeline gives you a **raster**: one class per pixel, or a probability per pixel. Most people want
something else — polygons they can open in a GIS, an area in hectares, an accuracy figure, a file that loads fast in
a web map. The postprocessing toolbox turns the raster into those products.

Everything here works on files you already have, so it is equally useful for output from `pygeovision infer`, from a
pipeline, or from another tool entirely. Nothing is downloaded.

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
post = client.postprocess          # pygeovision.data.postprocess.PostProcessor
```

## Operations at a glance

| Step | Method | CLI | Input → output |
|---|---|---|---|
| Raster → polygons | `vectorise` | `postprocess vectorise` | class GeoTIFF → GeoJSON, one feature per connected patch |
| Polygons → raster | `rasterise` | — | GeoJSON + reference GeoTIFF → GeoTIFF |
| Remove speckle | `sieve_filter` | `postprocess sieve` | class GeoTIFF → class GeoTIFF |
| Fill gaps | `fill_holes` | — | float GeoTIFF with nodata → filled GeoTIFF |
| Keep confident pixels | `apply_confidence_threshold` | — | probability GeoTIFF → 1/255 mask |
| Simplify outlines | `smooth` | `postprocess smooth` | GeoJSON → GeoJSON |
| Straighten buildings | `regularise_buildings` | `postprocess regularise` | GeoJSON → GeoJSON |
| Trim to an area | `clip_vector_to_polygon` | — | GeoJSON + boundary → GeoJSON |
| Merge touching polygons | `dissolve` | — | GeoJSON → GeoJSON |
| Grow or shrink | `buffer` | — | GeoJSON → GeoJSON |
| Area per class | `class_statistics` | `postprocess class-stats` | class GeoTIFF → dict |
| Statistics per polygon | `zonal_statistics` | `postprocess zonal-stats` | raster + GeoJSON → GeoJSON |
| Accuracy | `accuracy_assessment` | `postprocess accuracy` | prediction + reference → dict |
| Web-ready raster | `to_cog` | `postprocess cog` | GeoTIFF → Cloud-Optimized GeoTIFF |
| Other vector formats | `export` | `postprocess export` | GeoJSON → GPKG / Shapefile / KML |
| One-page summary | `generate_report` | — | prediction (+ reference) → HTML or JSON |

(units-and-crs)=
## Units: read this before choosing a threshold

Several operations take a size — a minimum area, a simplification tolerance, a buffer distance. What the number
means depends on the **coordinate system of the file**:

| Argument | Unit | Notes |
|---|---|---|
| `min_area_m2` (`vectorise`, `regularise_buildings`) | **square metres, always** | For a lon/lat file the true area on the WGS84 ellipsoid is used, not degrees² |
| `simplify_tolerance` (`vectorise`), `tolerance` (`smooth`) | **map units** | Metres for a projected file, **degrees** for a lon/lat file |
| `distance` (`buffer`), `buffer_px` (`dissolve`) | **map units** | Same rule |
| `area_m2` property written by `vectorise` and `dissolve` | square metres, always | True area in both cases |
| `class_statistics` areas | hectares / km² / m² | True cell area; see [Grids, pixels and units](../getting-started/concepts.md) |

```{warning}
**Degrees are not metres.** One degree of latitude is about 111 km, so a "tolerance of 0.5" means half a degree on a
lon/lat file and will collapse every polygon to nothing. For a lon/lat file one metre is roughly `0.00001`.
`smooth` logs a warning when it is given a value of 0.001 or more on a lon/lat file; `vectorise` and `class_statistics`
convert for you. When in doubt, reproject the raster to a metric CRS first
([Preprocessing toolbox](preprocessing.md)) and every number becomes metres.
```

How the toolbox decides a GeoJSON is in degrees: if the file carries a `crs` member (the one `vectorise` writes) that
decides. Otherwise coordinates that lie inside longitude/latitude range **and** span less than 5° are taken as
degrees; anything else is taken as metres.

## Raster to polygons: `vectorise`

**Why.** A class raster cannot be edited, joined to a table or overlaid on a cadastre. Polygons can. Each connected
patch of the same class becomes its own feature, so a district of 4 000 buildings opens in QGIS as 4 000 separate
shapes, each selectable and with its own area, not as one multipolygon.

**How.**

```python
post.vectorise(
    "prediction.tif", "buildings.geojson",
    target_class=1,          # only this class; omit to vectorise every class
    min_area_m2=25.0,        # drop patches smaller than 25 m²
    simplify_tolerance=0.0,  # optional Douglas–Peucker, in MAP UNITS
)
```

```bash
pygeovision postprocess vectorise prediction.tif --output buildings.geojson \
    --target-class 1 --min-area 25 --validate
```

| Argument | Default | Meaning |
|---|---|---|
| `band` | `1` | Band to read |
| `target_class` | `None` | Keep only pixels of this class |
| `min_area_m2` | `0` | Drop patches below this true area |
| `simplify_tolerance` | `0` | Douglas–Peucker tolerance in map units (0 = none) |
| `nodata` | file value | Override the nodata value; pixels equal to it are never vectorised |

**Expected output.** A `FeatureCollection`. Each feature has two properties:

```text
{"type": "Feature",
 "properties": {"class": 1, "area_m2": 294047.27},
 "geometry": {"type": "Polygon", "coordinates": [[ … ]]}}
```

The collection also carries a legacy `crs` member holding the raster's CRS, so tools that honour it place the
polygons correctly; GIS software reading the file as plain GeoJSON assumes WGS84, which is right for a lon/lat
raster. For a projected raster, **reproject the output** (`pygeovision postprocess export` or `ogr2ogr -t_srs
EPSG:4326`) before sharing it as GeoJSON.
`--validate` prints `Validation: PASSED` or the number of invalid geometries.

Without `target_class`, pixels with value `0` are background and never become features. Select `target_class=0` explicitly if you really want them. With `target_class` set, the `class` property is that class.

**What to do when…**

- *You get one enormous polygon instead of buildings* — the prediction merged neighbouring buildings into one patch.
  Vectorising cannot split them; use a model with an instance head, or the [OBIA](../obia/index.md) tools, or
  [`sieve`](#remove-speckle-sieve_filter) with a smaller `min_pixels` after cleaning the mask.
- *Millions of tiny polygons* — set `min_area_m2` to the smallest real object you care about (25 m² for buildings,
  1 000 m² for fields) and run `sieve` first.
- *Jagged "staircase" edges* — they follow the pixel grid by construction. `simplify_tolerance` removes
  vertices, `smooth` does the same on an existing file; for building footprints use
  [`regularise_buildings`](#straighten-buildings-regularise_buildings).
- *The polygons are shifted or tiny in the GIS* — the file is lon/lat but was read as metres, or the reverse. Check the
  CRS of the **raster** with `gdalinfo`, and see [Units](#units-and-crs).

## Polygons to raster: `rasterise`

```python
post.rasterise("labels.geojson", reference_path="scene.tif", output_path="labels.tif",
               burn_field="class", nodata=0, dtype="uint8")
```

The reference raster fixes extent, resolution and CRS; each polygon is burned with the value of `burn_field`
(`default_value` if the property is missing). Use it to turn hand-drawn labels into a training mask that lines up
with an image. The vector **must already be in the reference raster's CRS**; no reprojection is done.

## Remove speckle: `sieve_filter`

**Why.** Pixel classifiers produce isolated one- and two-pixel patches ("salt and pepper"). They inflate counts and
create thousands of useless polygons.

```python
post.sieve_filter("prediction.tif", min_pixels=10, output_path="sieved.tif", connectivity=4)
```

```bash
pygeovision postprocess sieve prediction.tif --min-pixels 10 --output sieved.tif
```

Patches with fewer than `min_pixels` connected pixels are merged into their largest neighbour (`connectivity` 4 or
8). Output is `int32` with the input's CRS and nodata.

```{note}
A patch size in pixels is a different physical area at every resolution: 10 pixels is 1 000 m² at 10 m but 9 000 m²
at 30 m. Choose `min_pixels = smallest real object ÷ pixel area`.
```

**Expected output.** On a synthetic prediction with a 3×3-pixel speck of class 1, class 1 drops from 3 909 to
3 900 pixels and the speck becomes background; larger patches are unchanged.

**Caveat.** The `nodata` value (255 in pipeline outputs) is sieved like any other class, so a small valid patch
surrounded by no-data can be absorbed into the no-data area. If your raster has large nodata regions, either use a
small `min_pixels` or vectorise first and filter by `min_area_m2`.

## Fill gaps: `fill_holes`

For **continuous** rasters (elevation, an index) with nodata gaps.

```python
post.fill_holes("dem.tif", output_path="dem_filled.tif", method="nearest")   # or "bilinear"
```

| `method` | How | Use for |
|---|---|---|
| `nearest` | GDAL inverse-distance fill from the nearest valid pixels, searching up to 100 pixels | Small gaps, smooth surfaces |
| `bilinear` | Linear interpolation over the valid pixels (SciPy); pixels outside the valid hull are filled from the nearest valid pixel | Larger or irregular gaps |

Output is `float32`. Do not use this on a class raster — interpolated class codes are meaningless. Filling invents
values: keep the original and mention it when you report results.

## Keep confident pixels: `apply_confidence_threshold`

```python
post.apply_confidence_threshold("water_prob.tif", "water_mask.tif", threshold=0.6)
```

Pixels with probability ≥ `threshold` become `1`; **all others become the nodata value (255), not 0**. The output
therefore says "confidently water" or "not confident", never "confidently dry". Make a second mask with the opposite
test if you need "confidently not water".

## Simplify outlines: `smooth`

```python
post.smooth("buildings.geojson", "smooth.geojson", tolerance=0.00001)   # ≈ 1 m on a lon/lat file
```

Despite its name this is **simplification** (Douglas–Peucker with topology preserved): it removes vertices that
deviate less than `tolerance` from the line. It does not round corners. It reduces file size and removes staircase
noise; it does not move vertices away from where the pixels were. Tolerance is in [map units](#units-and-crs).

## Straighten buildings: `regularise_buildings`

**Why.** Segmentation edges are slightly off-axis from pixel-grid noise even though real buildings are rectilinear.

```python
post.regularise_buildings("buildings.geojson", "regular.geojson",
                          angle_tolerance_deg=15.0, min_area_m2=10.0)
```

```bash
pygeovision postprocess regularise buildings.geojson --output regular.geojson --min-area 10
```

For every polygon it estimates the dominant orientation (from the minimum rotated rectangle — used only as an
orientation estimate, not as the output), rotates the building to axis alignment, snaps every edge that is within
`angle_tolerance_deg` of horizontal or vertical to exactly horizontal or vertical, and rotates back. L-shapes and
courtyards keep their shape; area is not inflated the way a bounding rectangle would. Features get
`"regularised": true`. Polygons below `min_area_m2` (true area) are dropped; a polygon that cannot be processed is
kept unchanged rather than lost.

**What to do when…** the result looks wrong — edges steeper than `angle_tolerance_deg` from the main directions are
left as they are, by design. Widen it (up to about 25°) for sloppy masks, narrow it for diagonal streets.
Only the exterior ring is processed: **holes (courtyards) are not straightened**.

## Trim to an area: `clip_vector_to_polygon`

```python
res = post.clip_vector_to_polygon("buildings.geojson", "accra_boundary.geojson", "accra_buildings.geojson")
print(res)   # {'output_path': '…', 'n_before': 4812, 'n_after': 4690}
```

The boundary may be a path, a Feature, a FeatureCollection or a bare geometry. It is assumed to be WGS84 and is
reprojected to the CRS of the vector being clipped. Features that cross the boundary are **cut** at it. Get a real
boundary with [`client.boundary()`](boundaries.md). Use this as the last step so that tile-edge artefacts and
overshoot beyond the study area are removed.

## Merge touching polygons: `dissolve`

```python
post.dissolve("patches.geojson", "merged.geojson", by_field="class", buffer_px=0.0)
```

Polygons of the same `by_field` value that touch or overlap are merged; `by_field=None` merges everything. A positive
`buffer_px` (map units) is applied first and removed afterwards, which bridges hairline gaps between neighbours.
Each result has `{by_field: value, area_m2}` with the true area. Typical use: turning pixel patches of one land-cover
class into contiguous land-cover polygons. **Do not dissolve buildings** if you need to count them.

## Grow or shrink: `buffer`

```python
post.buffer("roads.geojson", "road_corridor.geojson", distance=10.0)   # map units
```

Negative distances shrink. On a lon/lat file the distance is in degrees — reproject first to buffer in metres.

## Areas per class: `class_statistics`

**Why.** "How much of the map is each class?" is the first question anyone asks, and it must be answered in true
area, not in pixels.

```python
stats = post.class_statistics("prediction.tif")
for cls, s in stats.items():
    print(cls, s["pixels"], s["area_ha"], s["pct"])
```

```bash
pygeovision postprocess class-stats prediction.tif
```

**Expected output** (synthetic 200×200 raster, ≈11 m pixels, 10 rows of nodata):

```text
   Class       Pixels    Area (ha)     Area km²      Pct
  ────── ──────────── ──────────── ──────────── ────────
       0       33,691      412.786      4.12786   88.66%
       1        3,909       47.893      0.47893   10.29%
       2          400        4.901      0.04901    1.05%
```

The Python result is `{class_id: {"pixels", "area_m2", "area_ha", "area_km2", "pct"}}`. `pct` is a share of
**valid** (non-nodata) pixels. For a lon/lat raster each row has its own cell area (a degree of longitude shrinks
toward the poles); for a projected raster the area is `res_x × res_y`. Pass `pixel_size_m=` only to override this
with a square pixel of a given size. If the raster has **no nodata value set**, every pixel counts as valid.

## Statistics per polygon: `zonal_statistics`

**Why.** "What is the mean NDVI of each field?" "How confident is the model inside each parcel?"

```python
rows = post.zonal_statistics("ndvi.tif", "fields.geojson", stats=["mean", "std", "min", "max", "count"])
post.zonal_statistics("ndvi.tif", "fields.geojson", output_path="fields_ndvi.geojson")   # writes a file
```

```bash
pygeovision postprocess zonal-stats ndvi.tif fields.geojson --output fields_ndvi.geojson \
    --stats mean,std,min,max,count
```

Each zone keeps its original properties and gains `stat_mean`, `stat_std`, … Available statistics are `mean`,
`std`, `min`, `max`, `count`, `median`, `sum`. A zone with no valid pixels gets `null` for every statistic — not 0.

Zones are reprojected to the raster's CRS (a GeoJSON without a CRS member is assumed to be WGS84), nodata and NaN are
excluded, and integer rasters such as class maps are supported. `count` is the number of pixels, not an area.
Pixels are included when their centre lies inside the polygon.

## Accuracy against a reference: `accuracy_assessment`

**Why.** A map without an accuracy figure is an opinion. This compares a prediction with a reference raster of the
same grid and gives the standard metrics.

```python
report = post.accuracy_assessment("prediction.tif", "reference.tif",
                                  class_names=["Background", "Building", "Road"], ignore_class=None)
print(report["overall_accuracy"], report["kappa"], report["mean_iou"])
```

```bash
pygeovision postprocess accuracy prediction.tif reference.tif --class-names Background,Building,Road
```

**Expected output:**

```text
  Overall Accuracy : 0.9895
  Cohen's κ        : 0.9502
  Mean IoU         : 0.9651
  Total pixels     : 38,000

  Per-class metrics:
  Class  Name              Precision   Recall       F1      IoU
  ────── ──────────────── ────────── ──────── ──────── ────────
  0      Background           0.9881   1.0000   0.9940   0.9881
  1      Building             1.0000   0.9072   0.9513   0.9072
  2      Road                 1.0000   1.0000   1.0000   1.0000
```

The dictionary also holds `confusion_matrix` (rows = reference, columns = prediction, in sorted class order),
`class_names`, and per class `tp`, `fp`, `fn`. Pixels that are nodata in the reference *or* the prediction, and pixels
of `ignore_class`, are excluded.

```{warning}
The two rasters **must have the same shape and grid**. The function compares pixels in reading order and truncates to
the shorter one if sizes differ — it does not resample. A reference at another resolution or offset gives a
meaningless, but plausible-looking, number. Resample or clip the reference to the prediction first
([Preprocessing toolbox](preprocessing.md)).
```

Read the metrics together: overall accuracy is dominated by the largest class (a map that is 95 % background scores 0.95
by predicting nothing); **per-class recall and IoU** show whether you found the rare class; κ corrects for chance
agreement. Judge a result by the class you care about.

## Web-ready raster: `to_cog`

```python
post.to_cog("prediction.tif", "prediction_cog.tif")
```

Produces a Cloud-Optimized GeoTIFF: tiled, compressed, with overview pyramids (2, 4, 8, 16, 32), which web maps and
`rio`/QGIS read quickly without downloading the whole file. If overviews are missing from the engine's output they are
added. If neither the pygeofetch engine nor `gdal_translate` is available, a tiled GeoTIFF is written **without
overviews** and a warning says so.

## Other vector formats: `export`

```python
post.export("buildings.geojson", "buildings.gpkg", fmt="gpkg")   # geojson | shp | gpkg | kml
```

```bash
pygeovision postprocess export buildings.geojson --output buildings.gpkg --format gpkg
```

Prefer **GeoPackage**: no 10-character field-name limit, no 2 GB limit, one file, and it stores the CRS. Shapefile
truncates field names (`area_m2` survives, longer names do not) and writes a bundle of files. KML is for Google Earth.

## One-page summary: `generate_report`

```python
post.generate_report("prediction.tif", "summary.html", reference_path="reference.tif",
                     class_names=[...], fmt="html")      # or fmt="json"
```

A bare HTML table of class areas plus, when a reference is given, overall accuracy, κ and mIoU. For a full formatted
report (maps, methods, limitations) use the pipeline report builder instead: [Deliverables & reports](../reporting/index.md).

## A complete example: model output to a clean building layer

```python
post.sieve_filter("pred.tif", min_pixels=12, output_path="pred_sieved.tif")
post.vectorise("pred_sieved.tif", "b.geojson", target_class=1, min_area_m2=25)
post.regularise_buildings("b.geojson", "b_reg.geojson", min_area_m2=25)
post.clip_vector_to_polygon("b_reg.geojson", "boundary.geojson", "b_final.geojson")
post.export("b_final.geojson", "buildings.gpkg", fmt="gpkg")
print(post.class_statistics("pred_sieved.tif"))
```

## What to do when…

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: … rasterio / geopandas / shapely` | Optional geospatial extras missing | `pip install "pygeovision[geo]"` or install the named package |
| `vectorise` writes 0 features | No pixel has that `target_class`, the raster is all background (`0`) when no class is given, or all pixels equal nodata | Run `class-stats` and check which classes exist |
| Polygons vanish after `smooth` | Tolerance in degrees was far too large | Use ≈ `0.00001` per metre on a lon/lat file, or reproject first |
| `zonal_statistics` is all `null` | Zones do not overlap the raster, or they are in another CRS without a `crs` member and are not WGS84 | Check bounds with `gdalinfo` / `ogrinfo`; add a CRS to the vector |
| Accuracy is suspiciously high | Rasters on different grids, or most pixels are an easy background class | Check shapes match; read per-class IoU |
| COG has no overviews | Neither engine available | Install `pygeofetch` or GDAL, then rerun |
| Areas disagree with another tool | The other tool multiplies pixels by one pixel size on a lon/lat raster | This toolbox uses true cell area; for a metric CRS both agree |
