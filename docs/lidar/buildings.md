# Building footprints

```python
r = proc.extract_buildings("classified.las", "buildings.geojson", building_class=6,
                           resolution=0.5, min_area_m2=10.0, gap_fill_m=1.5, simplify_m=0.0)
```

**What it does.** Points of `building_class` (LAS 6) mark cells on a grid of `resolution`; a morphological **closing** joins
cells separated by gaps up to about `gap_fill_m` (LiDAR returns are scattered; without joining, one roof breaks into many
pieces); the mask is turned into polygons; polygons smaller than `min_area_m2` are dropped. `simplify_m > 0` smooths the
staircase edges (Douglas–Peucker tolerance in metres).

## Expected output

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
