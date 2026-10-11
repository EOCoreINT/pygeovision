"""
Microsoft Building Footprints labeler.

Accesses the Microsoft Building Footprints dataset (1.3 billion buildings globally)
and generates binary segmentation masks for satellite imagery tiles.

Dataset: https://github.com/microsoft/GlobalMLBuildingFootprints
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import BuildingFootprintError

logger = logging.getLogger(__name__)

# The partition index (dataset-links.csv) moves between releases, so it is located by
# pygeovision.data.footprints rather than pinned here.


class MicrosoftBuildingsLabeler(BaseLabeler):
    """
    Generate building segmentation labels using Microsoft Building Footprints.

    The Microsoft Global ML Building Footprints dataset covers 1.3 billion
    building footprints across the globe, derived from Bing Maps imagery
    using semantic segmentation AI.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching downloaded footprint data.
    confidence_threshold : float
        Minimum confidence for keeping a label.
    quadkey_zoom : int
        QuadKey zoom level for spatial indexing. Defaults to 9.

    Examples
    --------
    >>> labeler = MicrosoftBuildingsLabeler(cache_dir=Path("./ms_buildings_cache/"))
    >>> results = labeler.label_tiles(tiles=tiles, output_dir=Path("./labels/"))
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        confidence_threshold: float = 0.5,
        quadkey_zoom: int = 9,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".pygeovision_cache/ms_buildings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quadkey_zoom = quadkey_zoom

    @property
    def name(self) -> str:
        return "microsoft_buildings"

    @property
    def supported_tasks(self) -> list[str]:
        return ["segmentation", "detection"]

    def label_tile(
        self,
        tile: TileMetadata,
        output_dir: Path,
        task: str = "segmentation",
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Generate a building mask for a tile using Microsoft Building Footprints.

        Parameters
        ----------
        tile : TileMetadata
        output_dir : Path
        task : str
        **kwargs

        Returns
        -------
        LabelingResult
        """
        bounds_wgs84 = self._ensure_wgs84_bounds(tile)
        if bounds_wgs84 is None:
            return self._skip_tile(tile, "Cannot reproject bounds to WGS84")

        # The partition index, download and parsing live in pygeovision.data.footprints (shared with the
        # building_footprints pipeline); a failure is reported as the tile's error, never swallowed.
        from pygeovision.data import footprints as fp  # noqa: PLC0415

        try:
            gdf = fp.microsoft_footprints(bounds_wgs84, cache_dir=self.cache_dir)
        except fp.FootprintSourceError as exc:
            logger.warning("Microsoft footprints unavailable for tile %s: %s", tile.tile_id, exc)
            return LabelingResult(tile_id=tile.tile_id, source=self.name, error=str(exc))
        import shapely.geometry as sg  # noqa: PLC0415

        geometries = [sg.mapping(g) for g in gdf.geometry]

        if not geometries:
            return self._skip_tile(tile, "No building footprints found in tile bounds")

        # Rasterize footprints to mask
        try:
            mask = self._rasterize_buildings(
                geometries=geometries,
                bounds=bounds_wgs84,
                height=tile.height,
                width=tile.width,
            )
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Rasterization failed: {exc}",
            )

        if float(np.mean(mask > 0)) < 0.001:
            return self._skip_tile(tile, "No building pixels after rasterization")

        label_path = self._write_label_geotiff(mask, tile, output_dir)
        distribution = self._compute_class_distribution(
            mask, ["background", "building"]
        )

        return LabelingResult(
            tile_id=tile.tile_id,
            label_path=label_path,
            confidence=0.9,  # ML-derived buildings have high but imperfect accuracy
            source=self.name,
            class_distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_wgs84_bounds(
        self,
        tile: TileMetadata,
    ) -> tuple[float, float, float, float] | None:
        """Reproject tile bounds to WGS84 if necessary."""
        bounds = tile.bounds
        if tile.crs == "EPSG:4326":
            return bounds
        try:
            from pyproj import Transformer  # noqa: PLC0415

            t = Transformer.from_crs(tile.crs, "EPSG:4326", always_xy=True)
            min_x, min_y = t.transform(bounds[0], bounds[1])
            max_x, max_y = t.transform(bounds[2], bounds[3])
            return (min_x, min_y, max_x, max_y)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRS reprojection failed: %s", exc)
            return None

    @staticmethod
    def _rasterize_buildings(
        geometries: list[Any],
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Rasterize building footprints to binary mask."""
        try:
            from rasterio.features import rasterize  # noqa: PLC0415
            from rasterio.transform import from_bounds  # noqa: PLC0415
        except ImportError as exc:
            raise BuildingFootprintError(
                "rasterio is required. Install with: pip install rasterio"
            ) from exc

        min_lon, min_lat, max_lon, max_lat = bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        shapes = [(geom, 1) for geom in geometries]

        return rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
