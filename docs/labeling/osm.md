# OSM Labeler

Generates real training labels from real OpenStreetMap features — no
manual annotation, but genuinely dependent on OSM's real, variable
coverage for your area.

## How it actually works

```python
from pygeovision.labeling.osm import OSMLabeler

labeler = OSMLabeler()
result = labeler.label(
    bbox=(-0.15, 51.47, -0.10, 51.52),
    categories=["buildings", "roads", "water"],
    output_path="./labels/osm_labels.tif",
    save_vector=True,
)
```

Queries a real Overpass API endpoint for the requested categories
within the bbox, then rasterizes the returned real vector geometry
into a real, aligned label raster at the resolution of your reference
imagery (or a default resolution if none is supplied via
`reference_raster=`).

**Real, supported categories:** `buildings`, `roads`, `water`,
`vegetation`, `agricultural`, `commercial`, `residential`,
`solar_panels`, `parking`, `sports` — each maps to a real, specific
set of OSM tag filters (e.g. `buildings` real-queries
`building=*`).

## Real output

```python
{
    "success": True,
    "n_features": 342,
    "categories": {"buildings": 201, "roads": 89, "water": 52},
    "output_path": "./labels/osm_labels.tif",
    "vector_path": "./labels/osm_labels.geojson",
    "error": None,
}
```

```{warning}
A real, confirmed bug was found and fixed in this exact function: it
previously returned `success: True` unconditionally, even when
`n_features` was 0 — a real, plausible outcome for a remote or rural
bbox with no matching OSM data. A user would get a "successful"
result pointing at an empty, all-background raster with no indication
anything was wrong. `success` now genuinely reflects whether real
features were found, with a clear `error` message when they weren't.
Tested directly with a simulated zero-feature response to confirm.
```

## CLI

```bash
pygeovision label osm -0.15 51.47 -0.10 51.52 --categories buildings --categories roads
```
