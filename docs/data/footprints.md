# Building and road footprints

Building footprints and road lines are available as **vectors from published datasets**. They are not detections on your
imagery: they are as complete and as current as their source, and the date of the imagery behind them varies by place.
When you need a fresh detection on your own scene, train a segmentation network ([Training](../training/index.md)) and
run the `building_footprints` pipeline with `checkpoint=`.

## Sources

| Function | Source | Coverage | Notes |
|---|---|---|---|
| `microsoft_footprints(bbox)` | Microsoft Global ML Building Footprints | Global, by country/quadkey | Found through the project's `dataset-links.csv` index; cached locally |
| `google_footprints(bbox, min_confidence=0.65)` | Google Open Buildings v3 | Africa, South and South-East Asia, Latin America | One gzipped CSV per S2 level-4 cell; keeps `confidence ≥ min_confidence` |
| `osm_features(bbox, kind="building")` | OpenStreetMap via the Overpass API | Global, volunteer-mapped | Closed ways tagged `building`; multipolygon relations are **not** included |
| `osm_features(bbox, kind="highway")` | OpenStreetMap | Global | Road lines with `highway`, `name`, `surface`, `oneway` |

Every function returns a `geopandas.GeoDataFrame` in EPSG:4326 clipped to the box and **raises** `FootprintSourceError`
when the source is unreachable or does not cover the box. An empty frame means the source was read successfully and
really holds nothing there.

```python
from pygeovision.data.footprints import microsoft_footprints, google_footprints, osm_features

bbox = (-0.1500, 51.4700, -0.1000, 51.5200)
ms = microsoft_footprints(bbox)
go = google_footprints(bbox, min_confidence=0.7)
roads = osm_features(bbox, kind="highway")
print(len(ms), ms.geometry.area.sum())     # note: area in degrees² here; project to a metric CRS first
```

## Which source to use

| If you… | Use |
|---|---|
| Work in Africa/Asia/Latin America and want maximum completeness | Google Open Buildings (with a confidence threshold) |
| Work in Europe, North America or want consistent global ML coverage | Microsoft |
| Need attributes (name, levels, type) or recent additions | OpenStreetMap (completeness varies hugely by place) |
| Need to compare | Run two; counts differing by a factor of two or more is not unusual |

## Overpass reliability

The OSM route uses several public Overpass servers in turn, sends a proper `User-Agent`, and retries with back-off on
rate limits (HTTP 429) and refusals. A busy public server can still be slow; heavy use should run a private
Overpass instance (pass `endpoints=`).

## As pipelines

- [`building_footprints`](../pipelines/catalog/building_footprints.md) — writes a GeoPackage of every building as its own
  feature, plus a raster of building count per cell, a map and a report.
- [`road_network`](../pipelines/catalog/road_network.md) — length by road class, density per cell, a GeoPackage of
  every road.

The GeoJSON and GeoPackage hold **one feature per building**, so they open in QGIS/ArcGIS as individual shapes.

## Cleaning the vectors

[Postprocessing](postprocessing.md) has `regularise_buildings` (snap outlines to a dominant orientation),
`smooth`, `dissolve`, `clip_vector_to_polygon` and `export`.
