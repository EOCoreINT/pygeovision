# Vector layers

Every pipeline that produces classes also delivers them as **polygons**: one feature per connected patch, with the class, the area and a place to look at it, ready to open in a GIS. This
is what lets you click on an individual flooded field, burn scar or settlement, instead of staring at one raster.

## What you get

| Format | Files | Notes |
|---|---|---|
| **GeoPackage** (`gpkg`, default) | `vectors/<pipeline>.gpkg` | One file with all layers. No field-name limit. The best choice for GIS. |
| **GeoJSON** (`geojson`, default) | `vectors/geojson/<pipeline>_<layer>.geojson` | One file per layer. For web maps and quick sharing. |
| Shapefile (`shp`) | `vectors/shapefile/<pipeline>_<layer>.shp` | Legacy. Field names are cut to 10 characters. |
| KML (`kml`) | `vectors/kml/<pipeline>_<layer>.kml` | Google Earth. Needs a GDAL build with a KML writer; otherwise a warning says so. |

All vectors are in **EPSG:4326** (longitude/latitude). Choose with `--vector-formats gpkg,shp,kml` (or `vector_formats=[…]`), or `none` for no vector files.

### The GeoPackage layers

For a raster-derived layer called `flood_extent` the GeoPackage holds:

| Layer | Geometry | Contents |
|---|---|---|
| `aoi` | 1 polygon | The box you asked for, with the pipeline, title, period, area in km² and date generated. |
| `flood_extent` | one polygon per patch | The patches of the classes that mean something (see below). |
| `flood_extent_classes` | one multipolygon per class | The patches of each class merged, with total area and number of patches. Handy for a quick symbology or totals. |

The `_classes` layer is left out for very large layers (over 200,000 patches).

## Attributes

| Attribute | Meaning |
|---|---|
| `patch_id` | 1, 2, 3 … in order of decreasing area, so patch 1 is the biggest. Unique within a layer. |
| `class_id` | The raster value (for example 1 = flooded). |
| `class_name` | The name of the class (from the pipeline, or your `--class-names`). |
| `color_hex` | The colour used on the map, so your GIS can match it. |
| `area_ha`, `area_m2` | Area of the patch. |
| `pct_aoi` | The patch's share of your whole box, in per cent. |
| `cx`, `cy` | A point inside the patch (longitude, latitude), for labels and tables. |
| `pipeline`, `period` | Which analysis, which dates. |

**Areas are measured in an equal-area projection (EPSG:6933)**, not in degrees and not in a local UTM, so they are comparable between places and between runs.

### Which classes become polygons

Not every raster value is a feature. "Dry", "not water", "unburned" or "no change" would cover most of the map with one polygon, so the pipeline's report specification
lists the classes of interest. No-data (255) never becomes a polygon. Layers that a pipeline wrote itself (building footprints, roads, streams, fire points) are delivered with their own attributes plus `patch_id`,
`class_id`, `class_name`, `area_ha` and `pct_aoi` (for polygons), `cx`/`cy`, `pipeline` and `period`.

## How the polygons are made

- Pixels of one class that touch along an edge (4-connectivity) form one patch; pixels that touch only at a corner are separate patches.
- Outlines follow pixel edges, so they are **stair-stepped**. That is faithful to the raster. To smooth or square them, use [`postprocess smooth` / `regularise`](../data/postprocessing.md).
- Holes are kept as holes.

(size-limits)=
## Size limits

A noisy raster over a large area can produce millions of one-pixel patches. A layer is capped at **300,000 polygons**. If the raster would produce more, patches smaller than 4, 8, 16 … pixels
are merged into their neighbours with a sieve until the layer fits, and the layer's note, the report and the log say how many patches there were and what size was merged. The raster
and the **tables in the report are unaffected** (they count pixels, not polygons).

A pipeline-written vector with more than 300,000 features is not duplicated into the deliverables; the pipeline's own file still holds all of them, and the report says so.

If you need every patch or a different size limit, polygonise the raster yourself:

```bash
pygeovision postprocess vectorise flood_extent.tif --output flooded.geojson --target-class 1 --min-area 500
```

```python
from pygeovision.reporting import generate_deliverables
generate_deliverables(result, bbox, out, max_polygons=1_000_000)
```

## Using the layers

```python
import geopandas as gpd

flood = gpd.read_file("output/deliverables/vectors/flood_mapping.gpkg", layer="flood_extent")
flood.sort_values("area_ha", ascending=False).head(10)[["patch_id", "class_name", "area_ha", "pct_aoi"]]

# Which buildings are flooded?
buildings = gpd.read_file("buildings/deliverables/vectors/building_footprints.gpkg", layer="buildings")
exposed = gpd.sjoin(buildings, flood[flood.class_id == 1][["geometry"]], predicate="intersects")
print(len(exposed), "of", len(buildings), "buildings touch flooded patches")
```

In QGIS: *Layer → Add Layer → Add Vector Layer*, choose the `.gpkg`, tick the layers. To colour by class, use *Categorized* symbology on `class_name` and set the colour from `color_hex` with a data-defined override.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| "vector layer … has no features (no pixels of the selected classes)" | The result has none of the classes of interest, for example no flooding found | That is a valid result; check `valid_fraction` before concluding |
| Hundreds of thousands of tiny polygons | Pixel noise | Run `pygeovision postprocess sieve` on the raster, or vectorise with `--min-area` |
| Polygons look jagged | They follow pixels | `postprocess smooth` or `regularise` |
| Area in QGIS differs from `area_ha` | QGIS measures in the layer CRS (degrees) or on the ellipsoid with another setting | Use `area_ha`, or measure in EPSG:6933 |
| Shapefile attribute names are truncated | Shapefile limit | Use the GeoPackage |
| "KML export skipped: this GDAL build has no KML writer" | The installed GDAL cannot write KML | Use GeoJSON, or install a GDAL with LIBKML |
| `vector export 'gpkg' failed` | A GeoPackage with that name is open in another program | Close it and rebuild with `pygeovision report` |
| No vector files for a continuous result | Continuous rasters have no classes | Threshold first (`postprocess` / `indices`) or use the pipeline's class layer |
