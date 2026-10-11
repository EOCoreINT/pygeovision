"""Building footprints and roads as vectors, from three published sources.

* ``microsoft`` -- Microsoft Global ML Building Footprints (country/quadkey partitions, gzipped line-delimited
  GeoJSON, found through the project's ``dataset-links.csv`` index).
* ``google``    -- Google Open Buildings v3 (one gzipped CSV per S2 level-4 cell, found through the tile manifest).
* ``osm``       -- OpenStreetMap through the Overpass API (buildings and roads).

Every function returns a ``geopandas.GeoDataFrame`` in EPSG:4326 clipped to the box and **raises** when the source
cannot be reached or does not cover the box -- an empty frame is returned only when the source was read successfully
and really holds no feature there. The footprints are *published products* (machine-learning detections or
volunteered data), not a fresh detection on your imagery, and the date of the imagery behind them varies by place.

The network is injectable (``session`` with ``get``/``post``) so the parsing is tested offline on small fixtures.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Index of the Microsoft partitions, newest known first. The project moves it from time to time; when none of these
#: answers, the current link is read from the project's README (see :func:`_ms_discover_index`).
MS_INDEX_URLS = (
    "https://bfppub.blob.core.windows.net/%24web/2026-08-13/dataset-links.csv",
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
)
MS_README = "https://raw.githubusercontent.com/microsoft/GlobalMLBuildingFootprints/main/README.md"
MS_QUADKEY_ZOOM = 9
GOOGLE_TILES = "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
OVERPASS_ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                      "https://overpass.kumi.systems/api/interpreter", "https://overpass.openstreetmap.fr/api/interpreter")


class FootprintSourceError(RuntimeError):
    """The footprint source is unreachable, does not cover the box, or returned something unreadable."""


def _session(session: Any):
    if session is not None:
        return session
    import requests

    from pygeovision._version import __version__
    s = requests.Session()
    # Public servers (Overpass above all) refuse the default "python-requests" identity with HTTP 406, and ask clients
    # to say who they are.
    s.headers.update({"User-Agent": f"PyGeoVision/{__version__} (+https://github.com/pygeovision)", "Accept": "*/*"})
    return s


def _cache_dir(cache_dir: str | Path | None, name: str) -> Path:
    p = Path(cache_dir) if cache_dir else Path.home() / ".pygeovision" / "cache" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fetch_bytes(session: Any, url: str, timeout: float = 120.0, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - any HTTP-library error; re-raised below with the URL
            last = exc
            logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(min(2.0 ** attempt, 8.0))
    raise FootprintSourceError(f"could not download {url}: {last}")


def _bbox_check(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    west, south, east, north = (float(v) for v in bbox)
    if not (west < east and south < north):
        raise FootprintSourceError(f"bbox {tuple(bbox)} must be (min_lon, min_lat, max_lon, max_lat) with min < max")
    return west, south, east, north


def _frame(geoms: list[Any], rows: list[dict[str, Any]]):
    import geopandas as gpd
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


# ---------------------------------------------------------------------------------------------------------------------
# Quadkeys
# ---------------------------------------------------------------------------------------------------------------------

def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int) -> str:
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkeys_for_bbox(bbox: Sequence[float], zoom: int = MS_QUADKEY_ZOOM) -> list[str]:
    """Bing-tile quadkeys at ``zoom`` that cover the box."""
    west, south, east, north = _bbox_check(bbox)
    x0, y1 = _tile_xy(west, south, zoom)
    x1, y0 = _tile_xy(east, north, zoom)
    return [quadkey(x, y, zoom) for x in range(min(x0, x1), max(x0, x1) + 1) for y in range(min(y0, y1), max(y0, y1) + 1)]


# ---------------------------------------------------------------------------------------------------------------------
# Microsoft Global ML Building Footprints
# ---------------------------------------------------------------------------------------------------------------------

def _ms_discover_index(session: Any) -> str:
    """The current ``dataset-links.csv`` URL, read from the project README."""
    text = _fetch_bytes(session, MS_README, timeout=60.0).decode("utf-8", "replace")
    found = re.findall(r"https://[A-Za-z0-9.\-]+/(?:%24web|\$web)/[0-9]{4}-[0-9]{2}-[0-9]{2}/dataset-links\.csv", text)
    if not found:
        raise FootprintSourceError("no dataset-links.csv link found in the Microsoft footprints README")
    return sorted(found)[-1]


def _ms_index(session: Any, cache: Path, index_url: str | None, refresh: bool) -> list[dict[str, str]]:
    path = cache / "dataset-links.csv"
    if refresh or not path.exists():
        urls = [index_url] if index_url else list(MS_INDEX_URLS)
        errors: list[str] = []
        content: bytes | None = None
        for url in urls:
            try:
                content = _fetch_bytes(session, url, timeout=120.0, retries=2)
                break
            except FootprintSourceError as exc:
                errors.append(str(exc))
        if content is None and not index_url:
            try:
                content = _fetch_bytes(session, _ms_discover_index(session), timeout=120.0, retries=2)
            except FootprintSourceError as exc:
                errors.append(str(exc))
        if content is None:
            raise FootprintSourceError(
                "the Microsoft building-footprint index could not be downloaded:\n  " + "\n  ".join(errors)
                + "\nPass index_url=... with the current dataset-links.csv link from "
                  "https://github.com/microsoft/GlobalMLBuildingFootprints")
        path.write_bytes(content)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise FootprintSourceError(f"{path} is empty")
    missing = {"QuadKey", "Url"} - set(rows[0].keys())
    if missing:
        raise FootprintSourceError(f"{path} does not have the columns {sorted(missing)}; the index format has changed")
    return rows


def _iter_ms_lines(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise FootprintSourceError(f"{path}: line {n} is not JSON ({exc}); the file may be truncated") from exc


def microsoft_footprints(bbox: Sequence[float], *, cache_dir: str | Path | None = None, session: Any = None,
                         index_url: str | None = None, refresh_index: bool = False):
    """Microsoft ML building footprints intersecting ``bbox`` (GeoDataFrame, EPSG:4326)."""
    from shapely.geometry import box, shape

    west, south, east, north = _bbox_check(bbox)
    sess, cache = _session(session), _cache_dir(cache_dir, "ms_buildings")
    keys = set(quadkeys_for_bbox(bbox))
    rows = [r for r in _ms_index(sess, cache, index_url, refresh_index) if str(r["QuadKey"]).strip() in keys]
    if not rows:
        raise FootprintSourceError(
            "Microsoft building footprints do not cover this area (no partition for quadkeys "
            f"{sorted(keys)[:4]}...). Try source='google' or source='osm'.")
    aoi = box(west, south, east, north)
    geoms: list[Any] = []
    props: list[dict[str, Any]] = []
    for r in rows:
        url = r["Url"].strip()
        local = cache / (re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("//", 1)[-1])[-150:] + ".gz")
        if not local.exists():
            tmp = local.with_suffix(".part")
            tmp.write_bytes(_fetch_bytes(sess, url, timeout=600.0))
            tmp.replace(local)
        for feat in _iter_ms_lines(local):
            geom = feat.get("geometry")
            if not geom:
                continue
            g = shape(geom)
            if g.intersects(aoi):
                geoms.append(g)
                p = feat.get("properties") or {}
                props.append({"height_m": p.get("height") if (p.get("height") or -1) > 0 else None,
                              "confidence": p.get("confidence") if (p.get("confidence") or -1) >= 0 else None})
    logger.info("Microsoft footprints: %d buildings from %d partition file(s)", len(geoms), len(rows))
    return _frame(geoms, props)


# ---------------------------------------------------------------------------------------------------------------------
# Google Open Buildings v3
# ---------------------------------------------------------------------------------------------------------------------

def _google_tiles(session: Any, cache: Path, manifest_url: str) -> list[dict[str, Any]]:
    path = cache / "tiles.geojson"
    if not path.exists():
        path.write_bytes(_fetch_bytes(session, manifest_url, timeout=120.0))
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features")
    if not isinstance(feats, list) or not feats:
        raise FootprintSourceError(f"{path} is not a GeoJSON FeatureCollection with features")
    return feats


def google_footprints(bbox: Sequence[float], *, min_confidence: float = 0.65, cache_dir: str | Path | None = None,
                      session: Any = None, manifest_url: str = GOOGLE_TILES):
    """Google Open Buildings v3 polygons intersecting ``bbox`` with ``confidence >= min_confidence``."""
    import pandas as pd
    from shapely import wkt
    from shapely.geometry import box, shape

    west, south, east, north = _bbox_check(bbox)
    if not 0.0 <= min_confidence <= 1.0:
        raise FootprintSourceError(f"min_confidence must be between 0 and 1, got {min_confidence}")
    sess, cache = _session(session), _cache_dir(cache_dir, "google_buildings")
    aoi = box(west, south, east, north)
    urls = []
    for feat in _google_tiles(sess, cache, manifest_url):
        url = (feat.get("properties") or {}).get("tile_url")
        if url and feat.get("geometry") and shape(feat["geometry"]).intersects(aoi):
            urls.append(url)
    if not urls:
        raise FootprintSourceError("Google Open Buildings does not cover this area "
                                   "(it covers Africa, South and South-East Asia, Latin America and the Caribbean).")
    geoms: list[Any] = []
    props: list[dict[str, Any]] = []
    for url in urls:
        local = cache / re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("//", 1)[-1])[-150:]
        if not local.exists():
            tmp = local.with_suffix(".part")
            tmp.write_bytes(_fetch_bytes(sess, url, timeout=900.0))
            tmp.replace(local)
        opener = gzip.open if local.read_bytes()[:2] == b"\x1f\x8b" else open
        with opener(local, "rt", encoding="utf-8") as fh:
            for chunk in pd.read_csv(fh, chunksize=200_000):
                need = {"latitude", "longitude", "confidence", "geometry"}
                if not need <= set(chunk.columns):
                    raise FootprintSourceError(f"{local.name} lacks the columns {sorted(need - set(chunk.columns))}")
                sel = chunk[(chunk.longitude >= west - 0.01) & (chunk.longitude <= east + 0.01)
                            & (chunk.latitude >= south - 0.01) & (chunk.latitude <= north + 0.01)
                            & (chunk.confidence >= min_confidence)]
                for geom_wkt, conf, area in zip(sel.geometry, sel.confidence, sel.get("area_in_meters", [None] * len(sel))):
                    g = wkt.loads(geom_wkt)
                    if g.intersects(aoi):
                        geoms.append(g)
                        props.append({"confidence": float(conf), "area_m2_source": None if area is None else float(area)})
    logger.info("Google Open Buildings: %d buildings from %d file(s)", len(geoms), len(urls))
    return _frame(geoms, props)


# ---------------------------------------------------------------------------------------------------------------------
# OpenStreetMap (Overpass)
# ---------------------------------------------------------------------------------------------------------------------

def _overpass(session: Any, query: str, endpoints: Iterable[str], timeout: float) -> dict[str, Any]:
    """POST ``query`` to each mirror in turn. A busy server (HTTP 429 / 504) is given one more try after a short wait."""
    import time
    errors: list[str] = []
    for url in endpoints:
        body = None
        for attempt in (1, 2):
            try:
                resp = session.post(url, data={"data": query}, timeout=timeout)
                resp.raise_for_status()
                body = resp.json()
                break
            except Exception as exc:  # noqa: BLE001 - HTTP or JSON error; the next mirror is tried, all are reported
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 1 and code in (429, 504):
                    wait = getattr(getattr(exc, "response", None), "headers", {}) or {}
                    try:
                        delay = min(float(wait.get("Retry-After", 5)), 15.0)
                    except (TypeError, ValueError):
                        delay = 5.0
                    logger.info("%s is busy (HTTP %s); retrying in %.0f s", url, code, delay)
                    time.sleep(delay)
                    continue
                errors.append(f"{url}: {exc}")
                break
        if body is None:
            continue
        remark = str(body.get("remark", ""))
        if "runtime error" in remark or "out of memory" in remark.lower():
            errors.append(f"{url}: {remark}")
            continue
        return body
    raise FootprintSourceError("Overpass API request failed (the area may be too large; try a smaller bounding box):\n  "
                               + "\n  ".join(errors))


def osm_features(bbox: Sequence[float], kind: str = "building", *, session: Any = None, timeout: float = 180.0,
                 endpoints: Sequence[str] = OVERPASS_ENDPOINTS):
    """OpenStreetMap ways tagged ``building`` (polygons) or ``highway`` (lines) inside ``bbox``.

    Buildings drawn as multipolygon relations are not included (they are a small share and need relation assembly).
    """
    from shapely.geometry import LineString, Polygon

    west, south, east, north = _bbox_check(bbox)
    if kind not in ("building", "highway"):
        raise FootprintSourceError("kind must be 'building' or 'highway'")
    box_q = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    query = f'[out:json][timeout:{int(timeout)}];way["{kind}"]({box_q});out geom tags;'
    body = _overpass(_session(session), query, endpoints, timeout + 30)
    geoms: list[Any] = []
    props: list[dict[str, Any]] = []
    for el in body.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
        tags = el.get("tags") or {}
        if kind == "building":
            if len(pts) < 4 or pts[0] != pts[-1]:
                continue                                  # an unclosed way is not a footprint
            g = Polygon(pts)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            props.append({"osm_id": el.get("id"), "building": tags.get("building"), "name": tags.get("name"),
                          "levels": tags.get("building:levels")})
        else:
            if len(pts) < 2:
                continue
            g = LineString(pts)
            props.append({"osm_id": el.get("id"), "highway": tags.get("highway"), "name": tags.get("name"),
                          "surface": tags.get("surface"), "oneway": tags.get("oneway")})
        geoms.append(g)
    logger.info("OpenStreetMap: %d %s feature(s)", len(geoms), kind)
    return _frame(geoms, props)