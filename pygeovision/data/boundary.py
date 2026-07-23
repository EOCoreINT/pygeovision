"""
pygeovision.data.boundary
==========================

Fetch real administrative boundary polygons for a study area by name,
instead of hand-typing an assumed bounding box.

Uses OpenStreetMap's Nominatim API (``polygon_geojson=1``) to resolve a
place name to its actual boundary geometry, computes its true area in the
correct local UTM zone (auto-detected — no manual zone lookup needed), and
optionally validates the result against a known reference area so a
mis-resolved query (e.g. a point/POI instead of the intended administrative
area) is caught rather than silently used.

Usage::

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    aoi = client.boundary(
        "Accra Metropolitan District, Greater Accra Region, Ghana",
        reference_area_km2=60.0,
    )
    print(aoi.bbox, aoi.area_km2, aoi.utm_epsg)

    results = client.search(bbox=aoi.bbox, date_range=(...))
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def _require_shapely():
    try:
        from shapely.geometry import mapping, shape
        from shapely.ops import transform as shp_transform
        return shape, mapping, shp_transform
    except ImportError:
        raise ImportError("pip install shapely") from None


def utm_epsg_for(lon: float, lat: float) -> int:
    """Auto-detect the correct UTM zone EPSG code for a WGS84 coordinate.

    No manual "which zone is my city in" lookup needed — this is the
    standard 6°-wide zone formula, hemisphere-aware.
    """
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class AdminBoundary:
    """A resolved administrative boundary."""
    query: str
    display_name: str
    geojson: dict[str, Any]
    bbox: tuple[float, float, float, float]           # (lon_min, lat_min, lon_max, lat_max), WGS84
    area_km2: float
    utm_epsg: int
    osm_type: str | None = None
    osm_class: str | None = None
    reference_area_km2: float | None = None
    validated: bool = True
    geojson_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"AOI: {self.display_name}",
            f"  Query               : {self.query}",
            f"  OSM type / class    : {self.osm_type} / {self.osm_class}",
            f"  BBox (WGS84)        : {tuple(round(b, 4) for b in self.bbox)}",
            f"  Area (fetched)      : {self.area_km2:.1f} km²",
            f"  UTM zone (auto)     : EPSG:{self.utm_epsg}",
        ]
        if self.reference_area_km2 is not None:
            deviation = abs(self.area_km2 - self.reference_area_km2) / self.reference_area_km2
            status = "✓ passed" if self.validated else "✗ FAILED"
            lines.append(
                f"  Reference area check: {self.reference_area_km2:.1f} km² "
                f"({deviation:.1%} deviation) — {status}"
            )
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def fetch_admin_boundary(
    query: str,
    output_path: str | pathlib.Path | None = None,
    reference_area_km2: float | None = None,
    tolerance: float = 0.35,
    raise_on_mismatch: bool = True,
    timeout: int = 30,
) -> AdminBoundary:
    """Fetch a real administrative boundary polygon from OSM Nominatim.

    Args:
        query: Place name, e.g.
            ``"Accra Metropolitan District, Greater Accra Region, Ghana"``.
        output_path: If given, save the resolved GeoJSON Feature here.
        reference_area_km2: If given, cross-check the fetched polygon's
            true area (computed in the correct local UTM zone) against
            this known value, catching a wrong-entity resolution rather
            than silently proceeding with bad data.
        tolerance: Allowed fractional deviation from `reference_area_km2`
            before treating the result as suspect (OSM boundary detail/
            vintage naturally varies a bit — 0.35 = 35% by default).
        raise_on_mismatch: If True (default), raise ValueError when the
            reference check fails. If False, return the result anyway
            with `validated=False` and a warning attached.
        timeout: HTTP timeout in seconds.

    Returns:
        AdminBoundary with the geometry, bbox, true area, and
        auto-detected UTM EPSG code for this location.
    """
    import requests

    shape, mapping, shp_transform = _require_shapely()

    headers = {"User-Agent": "PyGeoVision/2.x (admin-boundary-fetch)"}
    params = {
        "q": query,
        "format": "geojson",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "limit": 5,
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    fc = resp.json()

    features = fc.get("features", [])
    polygons = [f for f in features
                if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
    if not polygons:
        got = [f["geometry"]["type"] for f in features] or ["<no results>"]
        raise RuntimeError(
            f"Nominatim returned no polygon geometry for '{query}' (got {got}). "
            f"Try a more specific query string."
        )

    # Prefer an administrative-boundary result over e.g. a matching POI
    # Nominatim's `format=geojson` output uses the property key `category`
    # (confirmed against the official API docs) — NOT `class`, which is only
    # used in the plain `format=json`/`jsonv2` outputs. Checking `class` here
    # would silently never match anything in geojson responses, defeating
    # the whole point of preferring the administrative-boundary result over
    # some other polygon match (e.g. a park, a place-boundary stub, etc).
    # Check both keys defensively in case Nominatim's schema shifts again.
    def _category(f):
        p = f.get("properties", {})
        return p.get("category") or p.get("class")

    admin = [f for f in polygons if _category(f) == "boundary"]
    best = (admin or polygons)[0]
    props = best.get("properties", {})

    geom_wgs84 = shape(best["geometry"])
    bbox = tuple(round(b, 6) for b in geom_wgs84.bounds)
    centroid = geom_wgs84.centroid
    utm_epsg = utm_epsg_for(centroid.x, centroid.y)

    import pyproj
    project = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True).transform
    geom_utm = shp_transform(project, geom_wgs84)
    area_km2 = geom_utm.area / 1_000_000

    warnings: list[str] = []
    validated = True
    if reference_area_km2 is not None:
        deviation = abs(area_km2 - reference_area_km2) / reference_area_km2
        if deviation > tolerance:
            validated = False
            msg = (
                f"Fetched boundary area ({area_km2:.1f} km²) deviates {deviation:.0%} "
                f"from the known reference ({reference_area_km2} km²) for '{query}'. "
                f"Two likely causes: (1) the query resolved to the wrong OSM entity — "
                f"check osm_type={props.get('osm_type')!r}, "
                f"category={(props.get('category') or props.get('class'))!r}, "
                f"display_name={props.get('display_name')!r} against what you expect; "
                f"or (2) this is the *right* entity, but OpenStreetMap's community-"
                f"digitized boundary for it is measurably smaller/larger than the "
                f"official statistic — a real, known limitation for many regions "
                f"(especially outside Europe/North America), not necessarily an error. "
                f"Inspect the geometry yourself (output_path, if given) before deciding "
                f"whether to proceed with raise_on_mismatch=False or a wider tolerance."
            )
            if raise_on_mismatch:
                raise ValueError(msg)
            warnings.append(msg)
            logger.warning(msg)

    geojson_path = None
    if output_path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(best, fh)
        geojson_path = str(output_path)

    return AdminBoundary(
        query=query,
        display_name=props.get("display_name", query),
        geojson=best,
        bbox=bbox,
        area_km2=round(area_km2, 3),
        utm_epsg=utm_epsg,
        osm_type=props.get("osm_type"),
        osm_class=props.get("category") or props.get("class"),
        reference_area_km2=reference_area_km2,
        validated=validated,
        geojson_path=geojson_path,
        warnings=warnings,
    )