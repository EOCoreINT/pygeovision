"""
Microsoft and Google Buildings Auto-Labelers (E1, E2).
Generate building footprint labels from global open building datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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

    def _fetch_buildings_geojson(self, bbox: tuple[float, ...]) -> dict:
        """Fetch MS buildings for the bbox via the dataset-links.csv manifest."""
        import gzip
        import json

        import requests

        try:
            manifest = self._load_manifest()
        except Exception as exc:
            logger.error("Failed to fetch MS Buildings manifest: %s", exc)
            return {"type": "FeatureCollection", "features": []}

        quadkeys = self._bbox_to_quadkeys(bbox, self.quadkey_zoom)
        # Manifest quadkeys may be at a different (often coarser) zoom than
        # our target; match on prefix in both directions so either a
        # manifest row's quadkey is a prefix of ours, or vice versa.
        matched_urls = set()
        manifest_quadkeys = manifest["QuadKey"].astype(str)
        for i, mqk in enumerate(manifest_quadkeys):
            for qk in quadkeys:
                if mqk.startswith(qk) or qk.startswith(mqk):
                    matched_urls.add(manifest["Url"].iloc[i])
                    break

        lon_min, lat_min, lon_max, lat_max = bbox
        features = []
        for url in list(matched_urls)[:20]:  # cap to avoid huge downloads
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code != 200:
                    continue
                raw = gzip.decompress(resp.content)
                # Line-delimited GeoJSON (one Feature-like JSON object per line)
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
                    conf = rec.get("properties", {}).get("confidence", 1.0)
                    if conf < self.min_confidence:
                        continue
                    # Cheap bbox pre-filter on the first coordinate
                    try:
                        coords = geom["coordinates"][0]
                        lon0, lat0 = coords[0][0], coords[0][1]
                        if not (lon_min - 0.1 <= lon0 <= lon_max + 0.1 and
                                lat_min - 0.1 <= lat0 <= lat_max + 0.1):
                            continue
                    except (KeyError, IndexError, TypeError):
                        pass
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"confidence": conf, "label_value": 1, "source": "Microsoft"},
                    })
            except Exception as exc:
                logger.debug("MS Buildings download failed for %s: %s", url, exc)

        if not features:
            logger.warning("No MS Buildings found for bbox=%s (confidence>=%.1f)", bbox, self.min_confidence)
        return {"type": "FeatureCollection", "features": features}

    def label(
        self,
        bbox: tuple[float, ...],
        output_path: str | Path = "./labels/ms_buildings.tif",
        reference_raster: str | None = None,
        resolution_m: float = 1.0,
        save_vector: bool = True,
    ) -> dict[str, Any]:
        """Generate a building label raster from Microsoft ML Buildings."""
        output_path = Path(output_path)
        geojson = self._fetch_buildings_geojson(bbox)
        n = len(geojson["features"])
        logger.info("Microsoft Buildings: %d footprints for bbox=%s", n, bbox)

        if save_vector:
            import json
            v = Path(str(output_path).replace(".tif", "_ms_buildings.geojson"))
            v.parent.mkdir(parents=True, exist_ok=True)
            with open(v, "w") as f:
                json.dump(geojson, f)

        # Rasterise buildings = 1
        self._rasterise_buildings(geojson, bbox, output_path, resolution_m, reference_raster)
        # Real fix, same pattern found and fixed in osm.py: this
        # previously returned success=True unconditionally, even with
        # n=0 real buildings found (a real, plausible outcome for a
        # rural/remote bbox).
        return {
            "success": n > 0, "n_buildings": n, "output_path": str(output_path), "source": "Microsoft",
            "error": None if n > 0 else f"0 real Microsoft building footprints found for bbox={bbox}.",
        }

    def _rasterise_buildings(self, geojson, bbox, output_path, resolution_m, reference_raster):
        try:
            import numpy as np
            import rasterio
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            from shapely.geometry import shape

            lon_min, lat_min, lon_max, lat_max = bbox
            if reference_raster:
                with rasterio.open(reference_raster) as ref:
                    transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs
            else:
                deg_per_m = 1/111320
                px = 1/(resolution_m * deg_per_m)
                width, height = int((lon_max-lon_min)*px), int((lat_max-lat_min)*px)
                transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
                import rasterio.crs; crs = rasterio.crs.CRS.from_epsg(4326)

            shapes = [(shape(f["geometry"]), 1) for f in geojson["features"] if f.get("geometry")]
            label = rasterize(shapes, out_shape=(height, width), transform=transform,
                              fill=0, dtype=np.uint8) if shapes else np.zeros((height, width), np.uint8)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(str(output_path), "w", driver="GTiff", height=height,
                                width=width, count=1, dtype="uint8", crs=crs,
                                transform=transform, compress="lzw") as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(source="Microsoft_ML_Buildings")
        except ImportError as exc:
            raise ImportError(f"rasterio + shapely required: {exc}")


class GoogleBuildingsLabeler:
    """Generate building labels from Google Open Buildings v3.

    1B+ building footprints across Africa, South/Southeast Asia, Oceania.
    Based on satellite imagery from 2016–2022.

    Example::

        labeler = GoogleBuildingsLabeler()
        result = labeler.label(
            bbox=(3.35, 6.45, 3.45, 6.55),   # Lagos, Nigeria
            output_path="./labels/google_buildings_lagos.tif",
        )
    """

    S2_CELL_URL = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/{cell_id}_buildings.csv.gz"

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence

    def _bbox_to_s2_cells(self, bbox: tuple[float, ...]) -> list[str]:
        """S2 cell TOKENS (not raw cell IDs) for the bbox at level 4.

        The Google Open Buildings GCS bucket names files by S2 *token*
        (a compact hex string, e.g. '23b') — not the raw 64-bit cell ID.
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
    ) -> dict[str, Any]:
        """Generate building labels from Google Open Buildings."""
        output_path = Path(output_path)
        cells = self._bbox_to_s2_cells(bbox)
        if not cells:
            return {"success": False, "error": "pip install s2sphere for Google Buildings"}

        import gzip
        import io

        import requests
        all_features = []
        for cell in cells[:4]:  # limit to 4 cells
            url = self.S2_CELL_URL.format(cell_id=cell)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                import pandas as pd
                from shapely import wkt as shapely_wkt
                from shapely.geometry import mapping as shapely_mapping
                df = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)))
                df = df[df["confidence"] >= self.min_confidence]
                for _, row in df.iterrows():
                    geom = shapely_mapping(shapely_wkt.loads(row["geometry"]))
                    all_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"confidence": row["confidence"], "label_value": 1, "source": "Google"},
                    })
            except Exception as exc:
                logger.debug("Google Buildings cell %s: %s", cell, exc)

        logger.info("Google Buildings: %d footprints", len(all_features))
        geojson = {"type": "FeatureCollection", "features": all_features}

        # Use MS-style rasterise
        ms = MicrosoftBuildingsLabeler()
        ms._rasterise_buildings(geojson, bbox, output_path, resolution_m, reference_raster)
        n_buildings = len(all_features)
        return {
            "success": n_buildings > 0, "n_buildings": n_buildings, "output_path": str(output_path), "source": "Google",
            "error": None if n_buildings > 0 else f"0 real Google building footprints found for bbox={bbox}.",
        }