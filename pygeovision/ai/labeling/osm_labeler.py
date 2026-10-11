"""
OpenStreetMap labeler.

Queries OSM via Overpass API to extract vector features (buildings, roads,
land use, water bodies, etc.) and rasterizes them to label masks aligned
with imagery tiles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import OSMLabelingError

logger = logging.getLogger(__name__)

# Default Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 60  # seconds

# What each ``feature_type`` paints: category -> raster value. Values come from the same table the
# area labeler (``pygeovision.labeling.osm``) uses, so geometry is handled identically (roads are buffered
# lines, lakes with islands are assembled, buildings win overlaps).
OSM_TILE_CLASSES: dict[str, dict[str, int]] = {
    "buildings": {"buildings": 1},
    "roads": {"roads": 1},
    "water": {"water": 1},
    "landuse": {"residential": 1, "commercial": 2, "agricultural": 3, "vegetation": 4},
    "all": {  # same numbers as OSMLabeler.label()
        "buildings": 1, "roads": 2, "water": 3, "vegetation": 4, "agricultural": 5,
        "commercial": 6, "residential": 7, "solar_panels": 8, "parking": 9, "sports": 10,
    },
}


class OSMLabeler(BaseLabeler):
    """
    Generate segmentation labels from OpenStreetMap data.

    Queries the Overpass API for the tile's bounding box and rasterizes
    the returned features to a label mask aligned with the tile.

    Parameters
    ----------
    feature_type : str
        OSM feature category to label:
        ``"buildings"``, ``"roads"``, ``"landuse"``, ``"water"``, ``"all"``.
    overpass_url : str
        Overpass API endpoint. Defaults to the public endpoint.
    request_delay : float
        Delay between API requests (seconds) to avoid rate limiting.
    confidence_threshold : float
        Minimum confidence for keeping a label. Defaults to 0.5.

    Examples
    --------
    >>> labeler = OSMLabeler(feature_type="buildings")
    >>> results = labeler.label_tiles(tiles=tile_list, output_dir=Path("./labels/"))
    """

    def __init__(
        self,
        feature_type: str = "buildings",
        overpass_url: str = OVERPASS_URL,
        request_delay: float = 1.0,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        if feature_type not in OSM_TILE_CLASSES:
            raise OSMLabelingError(
                f"Invalid feature_type '{feature_type}'. Choose from {set(OSM_TILE_CLASSES)}"
            )
        self.feature_type = feature_type
        self.overpass_url = overpass_url
        self.request_delay = request_delay
        self._last_request_time: float = 0.0

    @property
    def name(self) -> str:
        return "openstreetmap"

    @property
    def supported_tasks(self) -> list[str]:
        return ["segmentation", "detection", "classification"]

    def label_tile(
        self,
        tile: TileMetadata,
        output_dir: Path,
        task: str = "segmentation",
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Generate an OSM label mask for a single tile.

        Parameters
        ----------
        tile : TileMetadata
            Tile to label.
        output_dir : Path
            Where to write the label GeoTIFF.
        task : str
            Task type (affects which classes are included).
        **kwargs
            Additional parameters (``feature_type`` override, etc.).

        Returns
        -------
        LabelingResult
        """
        feature_type = kwargs.get("feature_type", self.feature_type)
        if feature_type not in OSM_TILE_CLASSES:
            raise OSMLabelingError(f"Invalid feature_type '{feature_type}'. Choose from {set(OSM_TILE_CLASSES)}")

        # Build bounding box (min_lon, min_lat, max_lon, max_lat)
        # Handle both legacy (bounds obj) and new TileMetadata (tuple)
        raw_bounds = tile.bounds
        if hasattr(raw_bounds, 'left'):
            bounds = (raw_bounds.left, raw_bounds.bottom, raw_bounds.right, raw_bounds.top)
        else:
            bounds = tuple(raw_bounds)
        tile_crs = tile.crs if isinstance(tile.crs, str) else str(tile.crs)
        if "4326" not in tile_crs:
            bounds = self._reproject_bounds(bounds, tile_crs, "EPSG:4326")

        # Rate limit
        self._throttle()

        classes = OSM_TILE_CLASSES[feature_type]
        try:
            geojson = self._fetch_geojson(bounds, feature_type)
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Overpass query failed: {exc}",
            )

        if not geojson["features"]:
            return self._skip_tile(tile, "No OSM features found in tile bounds")

        # Rasterize onto the tile's own grid (its CRS and transform), not onto a lon/lat envelope.
        try:
            mask = self._rasterize_geojson(geojson, tile, classes)
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Rasterization failed: {exc}",
            )

        # Filter by coverage
        positive_fraction = float(np.mean(mask > 0))
        if positive_fraction < 0.001:
            return self._skip_tile(tile, f"Positive pixel fraction too low: {positive_fraction:.4f}")

        # Write label GeoTIFF
        label_path = self._write_label_geotiff(mask, tile, output_dir)

        class_names = self._get_class_names(feature_type)
        distribution = self._compute_class_distribution(mask, class_names)

        return LabelingResult(
            tile_id=tile.tile_id,
            label_path=label_path,
            confidence=1.0,  # OSM data is authoritative
            source=self.name,
            class_distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Overpass API
    # ------------------------------------------------------------------

    def _fetch_geojson(
        self,
        bounds: tuple[float, float, float, float],
        feature_type: str,
    ) -> dict[str, Any]:
        """Query Overpass for ``bounds`` (lon_min, lat_min, lon_max, lat_max) and return categorised GeoJSON."""
        from pygeovision.labeling.osm import OSMLabeler as AreaLabeler  # noqa: PLC0415

        area = AreaLabeler(endpoint=self.overpass_url, timeout=OVERPASS_TIMEOUT)
        categories = list(OSM_TILE_CLASSES[feature_type])
        data = area._fetch_overpass(area._build_overpass_query(bounds, categories))
        if data is None:
            raise OSMLabelingError(f"Overpass request failed ({area._last_error or 'no response'})")
        remark = str(data.get("remark", ""))
        if "runtime error" in remark.lower():
            raise OSMLabelingError(f"Overpass could not complete the query: {remark}")
        return area._osm_to_geojson(data, categories)

    @staticmethod
    def _rasterize_geojson(geojson: dict[str, Any], tile: TileMetadata, classes: dict[str, int]) -> np.ndarray:
        """Burn categorised lon/lat features onto ``tile``'s exact pixel grid."""
        from affine import Affine  # noqa: PLC0415

        from pygeovision.labeling.osm import OSMLabeler as AreaLabeler  # noqa: PLC0415

        transform = Affine(*tile.transform) if len(tile.transform) == 6 else Affine.identity()
        crs = tile.crs if isinstance(tile.crs, str) else str(tile.crs)
        import rasterio.crs  # noqa: PLC0415

        label, _ = AreaLabeler()._burn(
            geojson, transform, tile.width, tile.height, rasterio.crs.CRS.from_user_input(crs),
            all_touched=False, value_map=classes,
        )
        return label

    def _throttle(self) -> None:
        """Enforce rate limiting between API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _rasterize_features(
        self,
        features: list[dict[str, Any]],
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
        feature_type: str,
    ) -> np.ndarray:
        """
        Rasterize OSM features to a label mask.

        Parameters
        ----------
        features : list of dict
            OSM elements returned by Overpass.
        bounds : tuple
            (min_lon, min_lat, max_lon, max_lat).
        height, width : int
            Output mask dimensions.
        feature_type : str
            Determines class assignment.

        Returns
        -------
        np.ndarray
            Label mask of shape (H, W) with integer class IDs.
        """
        try:
            import shapely.geometry as sg  # noqa: PLC0415
            from rasterio.features import rasterize as rio_rasterize  # noqa: PLC0415
            from rasterio.transform import from_bounds  # noqa: PLC0415
        except ImportError as exc:
            raise OSMLabelingError(
                "rasterio and shapely are required for OSM rasterization. "
                "Install with: pip install rasterio shapely"
            ) from exc

        min_lon, min_lat, max_lon, max_lat = bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Build node lookup for way reconstruction
        nodes: dict[int, tuple[float, float]] = {}
        for elem in features:
            if elem["type"] == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])

        shapes: list[tuple[Any, int]] = []
        for elem in features:
            if elem["type"] != "way":
                continue
            coords = [nodes[nid] for nid in elem.get("nd", []) if nid in nodes]
            if len(coords) < 3:
                continue

            try:
                geom = sg.Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:  # noqa: BLE001
                continue

            class_id = self._get_class_id(elem.get("tags", {}), feature_type)
            shapes.append((sg.mapping(geom), class_id))

        if not shapes:
            return np.zeros((height, width), dtype=np.uint8)

        mask = rio_rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        return mask

    @staticmethod
    def _get_class_names(feature_type: str) -> list[str]:
        """Class names for a feature type, index = raster value (0 is background)."""
        classes = OSM_TILE_CLASSES[feature_type]
        names = ["background"] + [""] * max(classes.values())
        singular = {"buildings": "building", "roads": "road"}
        for cat, value in classes.items():
            names[value] = singular.get(cat, cat) if len(classes) == 1 else cat
        return names

    @staticmethod
    def _reproject_bounds(
        bounds: tuple[float, float, float, float],
        src_crs: str,
        dst_crs: str,
    ) -> tuple[float, float, float, float]:
        """Reproject bounding box from src_crs to dst_crs."""
        try:
            from pyproj import Transformer  # noqa: PLC0415

            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            min_x, min_y = transformer.transform(bounds[0], bounds[1])
            max_x, max_y = transformer.transform(bounds[2], bounds[3])
            return (min_x, min_y, max_x, max_y)
        except ImportError:
            logger.warning("pyproj not installed — using bounds as-is (may be wrong CRS)")
            return bounds
