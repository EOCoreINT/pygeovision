# Point clouds (LiDAR)

**Package:** `pygeovision.advanced.pointcloud` · **Class:** `PointCloudProcessor` · **Needs:** `laspy` (`pip install laspy`; add
`lazrs` for `.laz`), `rasterio`, `scipy`; `shapely` and `pyproj` for building outlines; `torch` for `classify_points`

## Why

An airborne LiDAR survey is a list of millions of 3-D points, each with a height, an intensity and (if someone classified the
file) a class: ground, vegetation, building, water. Maps need rasters and polygons. `PointCloudProcessor` does the three
conversions that come up most:

| Method | Turns | Into | Typical use |
|---|---|---|---|
| `read` | LAS/LAZ | NumPy arrays | Inspect, or build your own analysis |
| `canopy_height_model` | classified points | A height-above-ground raster (CHM) | Forest height and cover, biomass inputs |
| `classify_points` | points | Points with a class, using a network **you trained** | Label an unclassified survey |
| `extract_buildings` | points classified as buildings | Footprint polygons (GeoJSON) | Building inventories, change monitoring |

```{admonition} What is not here
:class: warning
* **No pretrained LiDAR classifier ships with PyGeoVision**, and none exists that suits every survey. `classify_points` runs
  PointNet++ or RandLA-Net with weights *you* provide. There is also no point-cloud training loop in `Trainer`; you write one
  with `get_model('randlanet' | 'pointnet2-ssg', ...)`. Without a checkpoint the labels are random and the result says so.
* **No ground classification.** `canopy_height_model` expects ground to be classified (LAS class 2). Most delivered surveys
  already are; if yours is not, use `filter_ground=False` for a rough answer, or classify ground with a dedicated tool (PDAL,
  LAStools, CloudCompare) first.
* **Building outlines are raster-based.** They follow the grid and have staircase edges; they are not surveyed or squared
  footprints.
```

## How to use

```{literalinclude} ../_examples/advanced/pointcloud_demo.py
:language: python
:caption: pointcloud_demo.py
:linenos:
```

{download}`Download pointcloud_demo.py <../_examples/advanced/pointcloud_demo.py>`. It writes a real LAS file (a 100 m tile, 5 % slope, 40
trees of 8–20 m, one 20 × 12 m building 9 m tall, two stray 400 m returns tagged as noise; about 14 000 points) and runs every
method on it. Runs in seconds.

### Reading a file

```python
from pygeovision.advanced.pointcloud import PointCloudProcessor
proc = PointCloudProcessor()
pc = proc.read("tile.las")        # dict: n_points, x, y, z, intensity, classification, crs, bounds
```

`x, y, z` are float64 in the file's units; `classification` and `intensity` are the raw integer arrays (`intensity` is `None` if
the point format has none); `crs` is a string like `"EPSG:32630"` or `"unknown"`; `bounds` is `(xmin, ymin, xmax, ymax)`.
Errors are exceptions: `FileNotFoundError`, `ValueError` for a file that is not LAS/LAZ or has no points (for `.laz`
install `lazrs`), `ImportError` when laspy is missing. (Earlier versions returned `{"error": ...}`; check `try/except`
in old code.)

## What to do when…

| Situation | What to do |
|---|---|
| `ImportError: laspy is required` | `pip install laspy`; for `.laz` also `pip install lazrs` |
| `ValueError: could not read … as LAS/LAZ` | The file is damaged or compressed (`.laz`) without `lazrs`/`laszip` installed |
| CHM raises "has no ground points" | Classify ground first (PDAL `filters.smrf`, LAStools, CloudCompare CSF), or use `filter_ground=False` and treat the result as a minimum |
| CHM has a huge maximum | A stray return: keep `exclude_noise=True`, classify noise in your software, and report a high percentile instead of the maximum |
| Many nodata cells in the CHM | Too few points per cell. Increase `resolution` (2–5 m for sparse surveys) |
| CHM heights are too low under dense forest | Few or no ground returns under canopy: the nearest-neighbour DTM comes from clearings. Use a survey with a higher ground-point density |
| The GeoTIFF has no CRS | The LAS header has none. Set it with `gdal_edit.py -a_srs EPSG:xxxx` once you know it |
| `extract_buildings` returns "No points with classification=6" | The file has no building class. Run a classifier (yours via `classify_points`, then `class_codes=[…, 6]`), or pass the class id your file uses |
| Buildings merged / fragmented | Lower / raise `gap_fill_m`; check `resolution` (0.5–1 m is typical) |
| `ValueError: … geographic CRS` | Reproject the point cloud to a metric CRS (PDAL `filters.reprojection`) |
| `ValueError: checkpoint … does not fit a randlanet with num_classes=…` | The checkpoint was trained with another architecture, class count or input width. Match `model`, `num_classes` and the 4 input channels (xyz + intensity) |
| `ValueError: class_codes must be within 0..31` | Old LAS point formats (0–5) store the class in 5 bits; use codes ≤ 31 or write LAS 1.4 (formats 6+) |
| Run out of memory in `classify_points` | Lower `max_points` (4 096), or split the file first |
| Labels flip between runs | Keep `seed` fixed; on GPU, ties/atomic operations can still differ slightly |

## Verified here

Real LAS files were written and read with `laspy` 2.x: grid exactness (cell size equal to `resolution`; tree height landing on
the right cell on an off-multiple extent), heights below sea level, noise classes, nodata cells, missing ground, building
areas and edge handling, CRS output, geographic-CRS refusal, blockwise classification (every point predicted once, repeatable
with a seed, checkpoint loading and mismatch, class codes, files without intensity and tiny clouds). The
trained-classifier path was checked as plumbing only; **no accuracy claim is made for any LiDAR classification model**.

```{toctree}
:hidden:
:maxdepth: 1

models
canopy-height
buildings
classify
```
