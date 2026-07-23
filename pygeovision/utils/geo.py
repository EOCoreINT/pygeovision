"""
pygeovision.utils.geo
=====================
AOI / geometry helpers.

Replaces this pattern that appears in every notebook::

    coords  = MY_AOI["coordinates"][0]
    MY_BBOX = (
        min(c[0] for c in coords),
        min(c[1] for c in coords),
        max(c[0] for c in coords),
        max(c[1] for c in coords),
    )
    AOI_PATH = OUTPUT_DIR / "my_aoi.geojson"
    with open(AOI_PATH, "w") as f:
        json.dump({"type":"FeatureCollection","features":[
            {"type":"Feature","geometry":MY_AOI,"properties":{}}]}, f)

After::

    from pygeovision.utils import aoi_bbox, save_aoi_geojson

    BBOX     = aoi_bbox(MY_AOI)
    AOI_PATH = save_aoi_geojson(MY_AOI, OUTPUT_DIR / "my_aoi.geojson",
                                 name="My Study Area", area_km2=612)
"""
from __future__ import annotations

import json
import pathlib


def aoi_bbox(
    geojson: dict,
) -> tuple[float, float, float, float]:
    """
    Extract ``(lon_min, lat_min, lon_max, lat_max)`` from a GeoJSON geometry
    or FeatureCollection.

    Handles Polygon, MultiPolygon, Feature, and FeatureCollection types.

    Args:
        geojson: GeoJSON dict with ``"type"`` key.

    Returns:
        ``(lon_min, lat_min, lon_max, lat_max)`` bounding box in WGS84.

    Example::

        from pygeovision.utils import aoi_bbox

        ANKASA_AOI = {"type":"Polygon","coordinates":[[
            [-2.85, 5.15], [-2.55, 5.15], [-2.55, 5.38], [-2.85, 5.38], [-2.85, 5.15]
        ]]}
        bbox = aoi_bbox(ANKASA_AOI)
        # → (-2.85, 5.15, -2.55, 5.38)
    """
    def _collect_coords(obj):
        """Recursively collect all [lon, lat] coordinate pairs."""
        t = obj.get("type", "")
        if t == "FeatureCollection":
            coords = []
            for feat in obj.get("features", []):
                coords.extend(_collect_coords(feat))
            return coords
        if t == "Feature":
            return _collect_coords(obj.get("geometry", {}))
        if t == "Polygon":
            return obj["coordinates"][0]
        if t == "MultiPolygon":
            coords = []
            for poly in obj["coordinates"]:
                coords.extend(poly[0])
            return coords
        if t in ("Point", "MultiPoint", "LineString",
                 "MultiLineString", "GeometryCollection"):
            # Flatten whatever coordinate structure exists
            raw = obj.get("coordinates", [])
            if raw and isinstance(raw[0], (int, float)):
                return [raw]
            if raw and isinstance(raw[0], list) and isinstance(raw[0][0], (int, float)):
                return raw
            coords = []
            for item in raw:
                if isinstance(item, list) and isinstance(item[0], (int, float)):
                    coords.append(item)
                elif isinstance(item, list):
                    coords.extend(item)
            return coords
        return []

    coords = _collect_coords(geojson)
    if not coords:
        raise ValueError(
            f"Could not extract coordinates from GeoJSON type={geojson.get('type')!r}. "
            f"Pass a Polygon, MultiPolygon, Feature, or FeatureCollection."
        )
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def save_aoi_geojson(
    geojson:    dict,
    path:       str | pathlib.Path,
    name:       str | None = None,
    area_km2:   float | None = None,
    country:    str | None = None,
    **extra_properties,
) -> str:
    """
    Write an AOI GeoJSON geometry to a FeatureCollection file.

    Replaces the 5-line ``json.dump({"type":"FeatureCollection",...})`` block
    that appears in every notebook setup cell.

    Args:
        geojson:    GeoJSON geometry dict (Polygon, MultiPolygon, or Feature).
        path:       Output file path (``.geojson``).
        name:       ``"name"`` property value.
        area_km2:   ``"area_km2"`` property value.
        country:    ``"country"`` property value.
        **extra_properties: Any additional properties to embed.

    Returns:
        ``str(path)`` — the written file path.

    Example::

        from pygeovision.utils import save_aoi_geojson

        AOI_PATH = save_aoi_geojson(
            ANKASA_AOI,
            OUTPUT_DIR / "ankasa_aoi.geojson",
            name    = "Ankasa Conservation Area",
            area_km2= 612,
            country = "Ghana",
        )
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise geometry: if it's a Feature or FC, extract the geometry
    t = geojson.get("type", "")
    if t == "Feature":
        geometry = geojson.get("geometry", geojson)
    elif t == "FeatureCollection":
        feats = geojson.get("features", [])
        geometry = feats[0].get("geometry", geojson) if feats else geojson
    else:
        geometry = geojson

    props: dict = {}
    if name        is not None: props["name"]     = name
    if area_km2    is not None: props["area_km2"] = area_km2
    if country     is not None: props["country"]  = country
    props.update(extra_properties)

    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": geometry, "properties": props}
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    return str(path)


def bbox_area_km2(
    bbox: tuple[float, float, float, float],
) -> float:
    """
    Approximate area of a WGS84 bounding box in km².

    Uses the midpoint cosine correction for longitude width.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.

    Returns:
        Approximate area in km².
    """
    import math
    lon_min, lat_min, lon_max, lat_max = bbox
    mid_lat   = math.radians((lat_min + lat_max) / 2)
    km_per_deg_lon = 111.32 * math.cos(mid_lat)
    km_per_deg_lat = 110.574
    width_km  = (lon_max - lon_min) * km_per_deg_lon
    height_km = (lat_max - lat_min) * km_per_deg_lat
    return abs(width_km * height_km)
