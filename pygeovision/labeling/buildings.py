"""
Microsoft and Google Buildings auto-labelers.

Both produce a binary ``uint8`` raster (0 background, 1 building) plus, optionally, the footprints as
GeoJSON. Footprints are *model-derived from imagery* (not surveyed), so they carry the errors of the
model that made them; see ``docs/labeling/buildings.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sibling(path: Path, suffix: str) -> Path:
    """``<dir>/<stem><suffix>`` -- never equal to ``path`` itself, whatever its extension."""
    return path.with_name(path.stem + suffix)


def rasterise_footprints(geojson: dict, bbox, output_path: Path, resolution_m: float,
                         reference_raster: str | None, tag: str, all_touched: bool = False) -> None:
    """Burn polygon footprints (lon/lat) into a binary label raster.

    With ``reference_raster`` the output has exactly that grid and CRS (footprints are reprojected to
    it); without it the grid is lon/lat with about ``resolution_m`` square pixels. A grid that would
    exceed ``MAX_LABEL_PIXELS`` raises ``ValueError`` instead of exhausting memory.
    """
    try:
        import numpy as np
        import rasterio
        import rasterio.crs
        from rasterio.features import rasterize
        from rasterio.warp import transform_geom
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError(f"rasterio + shapely required: {exc}")
    from pygeovision.labeling.osm import OSMLabeler

    if reference_raster:
        with rasterio.open(reference_raster) as ref:
            transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs
    else:
        transform, width, height = OSMLabeler._grid_from_bbox(bbox, resolution_m)
        crs = rasterio.crs.CRS.from_epsg(4326)
    to_target = crs is not None and crs != rasterio.crs.CRS.from_epsg(4326)

    shapes = []
    for f in geojson["features"]:
        if not f.get("geometry"):
            continue
        g = f["geometry"]
        if to_target:
            g = transform_geom("EPSG:4326", crs, g)
        shapes.append((shape(g).__geo_interface__, 1))
    label = (rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8,
                       all_touched=all_touched)
             if shapes else np.zeros((height, width), np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", driver="GTiff", height=height, width=width, count=1,
                       dtype="uint8", crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(label[np.newaxis])
        dst.update_tags(source=tag, n_footprints=str(len(shapes)))


def _in_bbox(geom_dict: dict, bbox) -> bool:
    """True when the footprint really intersects the bbox (not just its first vertex)."""
    from shapely.geometry import box, shape
    try:
        return shape(geom_dict).intersects(box(*bbox))
    except Exception:
        return False


class MicrosoftBuildingsLabeler:
    """Generate building labels from Microsoft's Global ML Buildings Dataset.

    1.4B+ buildings globally, derived from Bing Maps imagery (Maxar, Airbus,
    Vexcel, IGN France). Distributed as a CSV manifest (`dataset-links.csv`)
    mapping (region, quadkey) -> a gzip-compressed line-delimited GeoJSON
    file. There is no simple tile-URL scheme — the actual per-quadkey
    filenames are essentially arbitrary (Spark output part-files), so the
    manifest must be consulted first to find the right download URL(s).

    Example::

        labeler = MicrosoftBuildingsLabeler()
        result = labeler.label(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            output_path="./labels/ms_buildings.tif",
            reference_raster="./data/sentinel2.tif",
        )
    """

    MANIFEST_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"

    def __init__(self, min_confidence: float = 0.5, quadkey_zoom: int = 9) -> None:
        self.min_confidence = min_confidence
        self.quadkey_zoom = quadkey_zoom
        self._manifest_cache: Any = None

    def _load_manifest(self):
        """Download (and cache) the dataset-links.csv manifest.

        Columns: Location, QuadKey, Url, Size, UploadDate.
        """
        if self._manifest_cache is not None:
            return self._manifest_cache
        import pandas as pd
        import requests
        resp = requests.get(self.MANIFEST_URL, timeout=60)
        resp.raise_for_status()
        import io
        self._manifest_cache = pd.read_csv(io.StringIO(resp.text), dtype=str)
        return self._manifest_cache

    @staticmethod
    def _bbox_to_quadkeys(bbox: tuple[float, ...], zoom: int) -> list[str]:
        """Bing Maps tile-system quadkeys covering the bbox at `zoom`."""
        import math
        lon_min, lat_min, lon_max, lat_max = bbox

        def deg2tile(lat, lon, z):
            lat = max(min(lat, 85.05112878), -85.05112878)
            n = 2 ** z
            x = int((lon + 180.0) / 360.0 * n)
            y = int((1.0 - math.log(math.tan(math.radians(lat)) +
                     1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
            return max(0, min(n - 1, x)), max(0, min(n - 1, y))

        def tile_to_quadkey(x, y, z):
            qk = []
            for i in range(z, 0, -1):
                digit = 0
                mask = 1 << (i - 1)
                if x & mask:
                    digit += 1
                if y & mask:
                    digit += 2
                qk.append(str(digit))
            return "".join(qk)

        x0, y0 = deg2tile(lat_max, lon_min, zoom)  # top-left
        x1, y1 = deg2tile(lat_min, lon_max, zoom)  # bottom-right
        quadkeys = []
        for tx in range(min(x0, x1), max(x0, x1) + 1):
            for ty in range(min(y0, y1), max(y0, y1) + 1):
                quadkeys.append(tile_to_quadkey(tx, ty, zoom))
        return quadkeys

    #: Most files one request may download. A bbox on a quadkey border can match more; the extra
    #: files are not fetched and ``truncated`` is set in the result.
    MAX_FILES = 20

    def _fetch_buildings_geojson(self, bbox: tuple[float, ...]) -> dict:
        """Fetch MS buildings for the bbox via the dataset-links.csv manifest.

        Footprints are kept when they intersect the bbox. A ``confidence`` below ``min_confidence``
        drops the footprint, but a negative confidence (the dataset's "not available" marker, used for
        many regions) keeps it. The collection carries ``n_files_matched``, ``n_files_read``,
        ``n_files_failed``, ``truncated`` and ``manifest_error``.
        """
        import gzip
        import json

        import requests

        meta = {"n_files_matched": 0, "n_files_read": 0, "n_files_failed": 0, "truncated": False,
                "manifest_error": None, "n_no_confidence": 0}
        empty = {"type": "FeatureCollection", "features": [], **meta}
        try:
            manifest = self._load_manifest()
        except Exception as exc:
            logger.error("Failed to fetch MS Buildings manifest: %s", exc)
            empty["manifest_error"] = f"{type(exc).__name__}: {exc}"
            return empty

        quadkeys = self._bbox_to_quadkeys(bbox, self.quadkey_zoom)
        # Manifest quadkeys may be at a different (often coarser) zoom than our target; match on
        # prefix in both directions.
        matched_urls: list[str] = []
        manifest_quadkeys = manifest["QuadKey"].astype(str)
        for i, mqk in enumerate(manifest_quadkeys):
            url = manifest["Url"].iloc[i]
            if url in matched_urls:
                continue
            if any(mqk.startswith(qk) or qk.startswith(mqk) for qk in quadkeys):
                matched_urls.append(url)
        meta["n_files_matched"] = len(matched_urls)
        if len(matched_urls) > self.MAX_FILES:
            meta["truncated"] = True
            logger.warning("%d building files match; reading only the first %d (set MAX_FILES to change).",
                           len(matched_urls), self.MAX_FILES)

        features = []
        for url in matched_urls[: self.MAX_FILES]:
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code != 200:
                    meta["n_files_failed"] += 1
                    continue
                raw = gzip.decompress(resp.content)
                meta["n_files_read"] += 1
                # Line-delimited GeoJSON (one Feature per line)
                for line in raw.decode("utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    geom = rec.get("geometry")
                    if not geom:
                        continue
                    conf = rec.get("properties", {}).get("confidence", -1.0)
                    try:
                        conf = float(conf)
                    except (TypeError, ValueError):
                        conf = -1.0
                    if conf < 0:
                        meta["n_no_confidence"] += 1
                    elif conf < self.min_confidence:
                        continue
                    if not _in_bbox(geom, bbox):
                        continue
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"confidence": conf, "label_value": 1, "source": "Microsoft"},
                    })
            except Exception as exc:
                meta["n_files_failed"] += 1
                logger.warning("MS Buildings download failed for %s: %s", url, exc)

        if not features:
            logger.warning("No MS Buildings found for bbox=%s (confidence>=%.1f)", bbox, self.min_confidence)
        return {"type": "FeatureCollection", "features": features, **meta}

    def label(
        self,
        bbox: tuple[float, ...],
        output_path: str | Path = "./labels/ms_buildings.tif",
        reference_raster: str | None = None,
        resolution_m: float = 1.0,
        save_vector: bool = True,
        all_touched: bool = False,
    ) -> dict[str, Any]:
        """Generate a building label raster (0 background, 1 building) from Microsoft ML Buildings.

        ``reference_raster`` makes the labels share that raster's grid and CRS (any CRS).
        ``success`` is False when the footprints could not be fetched or none lie in the bbox.
        The footprints are saved beside the raster as ``<stem>_ms_buildings.geojson``.
        """
        output_path = Path(output_path)
        geojson = self._fetch_buildings_geojson(bbox)
        n = len(geojson["features"])
        logger.info("Microsoft Buildings: %d footprints for bbox=%s", n, bbox)

        vector_path = None
        if save_vector:
            import json
            vector_path = _sibling(output_path, "_ms_buildings.geojson")
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vector_path, "w") as f:
                json.dump({k: v for k, v in geojson.items() if k in ("type", "features")}, f)

        rasterise_footprints(geojson, bbox, output_path, resolution_m, reference_raster,
                             "Microsoft_ML_Buildings", all_touched)
        if n > 0:
            error = None
        elif geojson.get("manifest_error"):
            error = f"could not read the Microsoft building manifest: {geojson['manifest_error']}"
        elif geojson.get("n_files_matched", 0) == 0:
            error = f"no Microsoft building files cover bbox={bbox} (the dataset has no coverage there)."
        elif geojson.get("n_files_read", 0) == 0:
            error = (f"{geojson['n_files_failed']} building file(s) matched bbox={bbox} but none could be "
                     "downloaded.")
        else:
            error = f"0 Microsoft building footprints found inside bbox={bbox}."
        return {
            "success": n > 0, "n_buildings": n, "output_path": str(output_path),
            "vector_path": str(vector_path) if vector_path else None, "source": "Microsoft",
            "n_files_matched": geojson.get("n_files_matched", 0),
            "n_files_read": geojson.get("n_files_read", 0),
            "n_files_failed": geojson.get("n_files_failed", 0),
            "truncated": geojson.get("truncated", False),
            "n_without_confidence": geojson.get("n_no_confidence", 0),
            "error": error,
        }

    def _rasterise_buildings(self, geojson, bbox, output_path, resolution_m, reference_raster):
        rasterise_footprints(geojson, bbox, Path(output_path), resolution_m, reference_raster,
                             "Microsoft_ML_Buildings")


class GoogleBuildingsLabeler:
    """Generate building labels from Google Open Buildings v3.

    ~1B footprints across Africa, South and Southeast Asia, Latin America and the Caribbean
    (not Europe, North America, Australia or China). Based on 2016-2022 satellite imagery.

    The data is published per S2 level-4 cell (hundreds of km across). A request downloads every
    cell that touches the bbox **in full** (often tens to hundreds of MB, compressed), then keeps the
    rows inside the bbox, so the first call for an area is slow even when the bbox is small.

    Example::

        labeler = GoogleBuildingsLabeler()
        result = labeler.label(
            bbox=(3.35, 6.45, 3.45, 6.55),   # Lagos, Nigeria
            output_path="./labels/google_buildings_lagos.tif",
        )
    """

    S2_CELL_URL = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/{cell_id}_buildings.csv.gz"
    #: Most cells one request may download (extras are skipped and ``truncated`` is set).
    MAX_CELLS = 4

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence

    def _bbox_to_s2_cells(self, bbox: tuple[float, ...]) -> list[str]:
        """S2 cell TOKENS (not raw cell IDs) for the bbox at level 4.

        The Google Open Buildings GCS bucket names files by S2 *token*
        (a compact hex string, e.g. '23b') -- not the raw 64-bit cell ID.
        """
        try:
            import s2sphere
            lon_min, lat_min, lon_max, lat_max = bbox
            region = s2sphere.LatLngRect(
                s2sphere.LatLng.from_degrees(lat_min, lon_min),
                s2sphere.LatLng.from_degrees(lat_max, lon_max),
            )
            coverer = s2sphere.RegionCoverer()
            coverer.min_level = coverer.max_level = 4
            covering = coverer.get_covering(region)
            return [c.to_token() for c in covering]
        except ImportError:
            logger.warning("s2sphere not installed; pip install s2sphere for Google Buildings")
            return []

    def label(
        self,
        bbox: tuple[float, ...],
        output_path: str | Path = "./labels/google_buildings.tif",
        reference_raster: str | None = None,
        resolution_m: float = 1.0,
        save_vector: bool = True,
        all_touched: bool = False,
    ) -> dict[str, Any]:
        """Generate building labels (0 background, 1 building) from Google Open Buildings.

        ``success`` is False when s2sphere is missing, no cell could be downloaded, or no footprint
        lies in the bbox (e.g. a region the dataset does not cover). Footprints are saved beside the
        raster as ``<stem>_google_buildings.geojson``.
        """
        output_path = Path(output_path)
        cells = self._bbox_to_s2_cells(bbox)
        if not cells:
            return {"success": False, "error": "pip install s2sphere for Google Buildings"}

        import gzip
        import io

        import pandas as pd
        import requests
        from shapely import wkt as shapely_wkt
        from shapely.geometry import mapping as shapely_mapping

        lon_min, lat_min, lon_max, lat_max = bbox
        truncated = len(cells) > self.MAX_CELLS
        if truncated:
            logger.warning("%d S2 cells cover the bbox; reading only the first %d.", len(cells), self.MAX_CELLS)
        all_features = []
        n_cells_read = n_cells_failed = 0
        for cell in cells[: self.MAX_CELLS]:
            url = self.S2_CELL_URL.format(cell_id=cell)
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code != 200:
                    n_cells_failed += 1
                    logger.warning("Google Buildings cell %s: HTTP %s (no coverage or unreachable)",
                                   cell, resp.status_code)
                    continue
                df = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)))
                n_cells_read += 1
                df = df[df["confidence"] >= self.min_confidence]
                if {"latitude", "longitude"} <= set(df.columns):     # cheap vectorised pre-filter
                    pad = 0.01
                    df = df[df["longitude"].between(lon_min - pad, lon_max + pad)
                            & df["latitude"].between(lat_min - pad, lat_max + pad)]
                for _, row in df.iterrows():
                    geom = shapely_mapping(shapely_wkt.loads(row["geometry"]))
                    if not _in_bbox(geom, bbox):
                        continue
                    all_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"confidence": float(row["confidence"]), "label_value": 1, "source": "Google"},
                    })
            except Exception as exc:
                n_cells_failed += 1
                logger.warning("Google Buildings cell %s failed: %s", cell, exc)

        logger.info("Google Buildings: %d footprints", len(all_features))
        geojson = {"type": "FeatureCollection", "features": all_features}

        vector_path = None
        if save_vector:
            import json
            vector_path = _sibling(output_path, "_google_buildings.geojson")
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            with open(vector_path, "w") as f:
                json.dump(geojson, f)

        rasterise_footprints(geojson, bbox, output_path, resolution_m, reference_raster,
                             "Google_Open_Buildings", all_touched)
        n_buildings = len(all_features)
        if n_buildings > 0:
            error = None
        elif n_cells_read == 0:
            error = (f"none of the {min(len(cells), self.MAX_CELLS)} S2 cell file(s) for bbox={bbox} could be "
                     "downloaded -- the dataset has no coverage here or the bucket is unreachable.")
        else:
            error = f"0 Google building footprints (confidence >= {self.min_confidence}) found inside bbox={bbox}."
        return {
            "success": n_buildings > 0, "n_buildings": n_buildings, "output_path": str(output_path),
            "vector_path": str(vector_path) if vector_path else None, "source": "Google",
            "n_cells_read": n_cells_read, "n_cells_failed": n_cells_failed, "truncated": truncated,
            "error": error,
        }
