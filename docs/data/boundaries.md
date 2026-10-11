# Study-area boundaries

A bounding box includes more than your study area (and a hand-typed one is often wrong). `client.boundary()` resolves
an **administrative area by name** to its real polygon, measures its true area in the right UTM zone, and optionally
checks that area against a figure you know — so a query that resolved to the wrong place is caught, not used.

## Usage

```python
client = pgv.PyGeoVision()
aoi = client.boundary(
    "Accra Metropolitan District, Greater Accra Region, Ghana",
    output_path="aoi/accra.geojson",       # optional: save the polygon
    reference_area_km2=KNOWN_AREA_KM2,     # optional: sanity check against an area you know (km²)
    tolerance=0.35,                        # allowed fractional deviation (default 35 %)
)
print(aoi.summary())
results = client.search(bbox=aoi.bbox, date_range=("2024-06-01", "2024-06-30"),
                        collections=["sentinel-2-l2a"], geometry_file="aoi/accra.geojson")
```

`summary()` prints one block like this (values come from OpenStreetMap at the moment you run it):

```text
AOI: <display name from Nominatim>
  Query               : <your query>
  OSM type / class    : relation / boundary
  BBox (WGS84)        : (<min_lon>, <min_lat>, <max_lon>, <max_lat>)
  Area (fetched)      : <area> km²
  UTM zone (auto)     : EPSG:<326xx or 327xx>
  Reference area check: <known> km² (<deviation>) — ✓ passed
```

`AdminBoundary` fields: `query`, `display_name`, `geojson` (a Feature), `bbox`, `area_km2`, `utm_epsg`, `osm_type`,
`osm_class`, `reference_area_km2`, `validated`, `geojson_path`, `warnings`.

## Behaviour

- Uses OpenStreetMap **Nominatim** (`polygon_geojson=1`). Respect its usage policy: one query per second, no bulk runs.
- The area is computed on the polygon projected to its own UTM zone (`utm_epsg_for(lon, lat)`; the standard
  6°-wide zones, hemisphere-aware).
- If you pass `reference_area_km2` and the area deviates by more than `tolerance`, `raise_on_mismatch=True`
  (default) raises `ValueError`; with `False` the result is returned with `validated=False` and a warning.
- Up to five Nominatim results are requested. Of those that have a polygon geometry, the first with category
  `boundary` is used, otherwise the first polygon. If no result has a polygon (the query matched a point or a
  street), a `RuntimeError` says what came back and asks for a more specific query.

## Use with pipelines

Pipelines take a **bounding box**. Use `aoi.bbox` for the box and keep the polygon for clipping outputs:

```python
result = client.pipeline("water_bodies", bbox=aoi.bbox, date="2024-06")
# clip the statistics to the real boundary
client.postprocess.zonal_statistics(str(result.output_path), "aoi/accra.geojson", stats=["mean", "count"])
```

## What to do when…

| You see | Do this |
|---|---|
| `ValueError: … area deviates` | The place name matched something else. Add the region and country; check the OSM name on openstreetmap.org |
| Several candidates | The first Nominatim hit is used; make the query more specific |
| Timeout | Raise `timeout=` or retry later; Nominatim can be busy |
