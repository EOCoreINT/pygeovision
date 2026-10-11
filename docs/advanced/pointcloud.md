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

```{literalinclude} _examples/pointcloud_demo.py
:language: python
:caption: pointcloud_demo.py
:linenos:
```

{download}`Download pointcloud_demo.py <_examples/pointcloud_demo.py>`. It writes a real LAS file (a 100 m tile, 5 % slope, 40
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

## Canopy height model

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

### Expected output

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

### No ground classification

```text
ValueError: .../unclassified.las has no ground points (LAS class 2), so no terrain model can be built. Classify the gro ...
filter_ground=False -> max_m 18.7 canopy_cover_2m 0.119
```

With `filter_ground=True` and no class-2 points the function **raises** (an earlier version warned "CHM will be all zeros" and
wrote absolute elevations instead). With `filter_ground=False` the lowest return in each cell stands in for the ground. The
estimate is lower than the truth under dense canopy (cover 0.119 versus 0.177 with real ground), because in a cell fully
covered by crowns the lowest return may be a branch. Treat it as a minimum.

## Building footprints

```python
r = proc.extract_buildings("classified.las", "buildings.geojson", building_class=6,
                           resolution=0.5, min_area_m2=10.0, gap_fill_m=1.5, simplify_m=0.0)
```

**What it does.** Points of `building_class` (LAS 6) mark cells on a grid of `resolution`; a morphological **closing** joins
cells separated by gaps up to about `gap_fill_m` (LiDAR returns are scattered; without joining, one roof breaks into many
pieces); the mask is turned into polygons; polygons smaller than `min_area_m2` are dropped. `simplify_m > 0` smooths the
staircase edges (Douglas–Peucker tolerance in metres).

### Expected output

```text
{'success': True, 'output_path': '.../b.geojson', 'n_buildings': 1, 'n_building_points': 1500, 'total_area_m2': 234.0}
area_m2: [234.0] | true roof 20 x 12 = 240 | crs urn:ogc:def:crs:EPSG::32630
{'success': False, 'error': 'No points with classification=9 found. Run classify_points() first if this file lacks building classification, or pass the correct class id.'}
```

* One building of true area 240 m² was found with 234 m² (−2.5 %), as a polygon with properties `area_m2` and `source`. The
  GeoJSON carries a `crs` member (`urn:ogc:def:crs:EPSG::32630`) when the file's CRS has an EPSG code; with no CRS it has no
  `crs` member, and the coordinates are in the file's units.
* A class with no points returns `success: False` with a message (nothing is written); it is not an exception.
* **Areas are measured in the file's CRS**, so it must be projected, in metres. A geographic CRS (EPSG:4326) raises
  `ValueError` asking you to reproject; a file with no CRS is assumed to be in metres (logged).
* Tuning: raise `gap_fill_m` for sparse clouds; lower it if neighbouring buildings merge. Buildings touching the tile edge
  are kept intact. Tile your survey and merge polygons afterwards if the area is large.

## Classifying points with a network you trained

```python
r = proc.classify_points(
    "tile.las", "classified.las", num_classes=3, class_names=["ground", "vegetation", "building"],
    model="randlanet", checkpoint_path="my_randlanet.pt",
    class_codes=[2, 5, 6],                      # write standard LAS codes
)
```

### How a cloud is fed to the network

Point-cloud networks cannot take a whole survey at once (their neighbour search grows with the square of the number of
points). `classify_points` therefore:

1. cuts the cloud into `block_size` × `block_size` squares (default 50 m);
2. splits a square with more than `max_points` points (8 192) at random into several chunks, and pads a chunk with fewer than
   `min_points` (1 024) by repeating points (the padding is discarded afterwards);
3. gives the network, per chunk, **four channels**: x, y, z minus the chunk's centroid, divided by `coord_scale`
   (default 1.0), and intensity divided by `intensity_scale` (default 65 535; zeros if the file has no intensity);
4. writes each point's predicted class into the LAS classification field of a copy of the file (original coordinates intact).
   Every point is predicted exactly once.

**Train with the same input convention**, or the network sees inputs it never learned. If you trained on unit-scaled blocks,
set `coord_scale` to match.

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `num_classes` | 3 | Must equal the checkpoint's output classes |
| `class_names` | `class_0…` | Names for the count report |
| `model` | `"randlanet"` | `"randlanet"` or `"pointnet2"` (PointNet++ SSG). Any other value raises `ValueError` |
| `checkpoint_path` | `None` | A `state_dict` saved with `torch.save` (a dict with a `state_dict` / `model_state_dict` key and `module.` prefixes are accepted). Loaded with `weights_only=True` |
| `class_codes` | class index | LAS code written for each class: map to 2 ground, 3–5 vegetation, 6 building … so other software understands the file. Must be 0–31 for LAS point formats 0–5, 0–255 above |
| `block_size`, `max_points`, `min_points` | 50.0, 8192, 1024 | Chunking (above) |
| `coord_scale`, `intensity_scale` | 1.0, 65535 | Input scaling (above) |
| `device` | auto | |
| `seed` | 0 | Fixes the networks' random sampling and the chunk split, so a run is repeatable |

### Expected output without a trained model

```text
{'success': True, 'n_points': 13902, 'model': 'randlanet', 'used_pretrained_checkpoint': False, 'meaningful': False,
 'n_chunks': 4, 'class_counts': {'ground': 0, 'vegetation': 6562, 'building': 7340}}
warning: no checkpoint_path: the network has random weights, so the classes are meaningless (plumbing test only). ...
```

Look at `meaningful: False` and the `warning` key: a random network splits the points between classes arbitrarily (here 47 %
vegetation, 53 % building, for a scene that is mostly ground). This mode exists to test your file handling and output only.

With a checkpoint, `meaningful` is `True`, there is no `warning`, and `used_pretrained_checkpoint` is `True` (the name is
historical: it means "a checkpoint was loaded").

### Cost

Measured on a 2-core CPU: 300 000 points took about **29 s** with PointNet++ and **35 s** with RandLA-Net (64 chunks), peaking
below 1 GB of memory. A GPU is much faster; a real survey of tens of millions of points should be split into files first.

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
