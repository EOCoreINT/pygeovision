"""
Google Open Buildings Labeler.

Generates building segmentation labels using the Google Open Buildings dataset,
which contains ~1.8 billion building footprints across Africa, South Asia,
Southeast Asia, Latin America, and the Caribbean.

Example:
    >>> from pygeovision.ai.labeling.google_buildings import GoogleBuildingsLabeler
    >>> labeler = GoogleBuildingsLabeler(confidence_threshold=0.7)
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# Google Open Buildings dataset hosted on Google Cloud Storage
_GOB_BASE_URL = (
    "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip"
)

# S2 cell ID lookup endpoint
_S2_LOOKUP_URL = "https://storage.googleapis.com/open-buildings-data/v3/tiles.csv.gz"


@dataclass
class GoogleBuildingsConfig:
    """Configuration for the Google Open Buildings labeler.

    Attributes:
        confidence_threshold: Minimum confidence score (0-1) for including buildings.
        building_value: Pixel value assigned to building pixels.
        background_value: Pixel value assigned to non-building pixels.
        cache_dir: Directory for caching downloaded tile files.
        request_timeout: HTTP request timeout in seconds.
        max_workers: Maximum number of parallel download workers.
    """

    confidence_threshold: float = 0.65
    building_value: int = 1
    background_value: int = 0
    cache_dir: Path | None = None
    request_timeout: int = 60
    max_workers: int = 4


class GoogleBuildingsLabeler(BaseLabeler):
    """Labeler using Google Open Buildings dataset (~1.8 billion footprints).

    Queries the publicly available Google Open Buildings dataset to generate
    building segmentation masks aligned with satellite imagery tiles.

    Coverage: Africa, South Asia, Southeast Asia, Latin America, Caribbean.

    Args:
        confidence_threshold: Minimum building confidence score (0.0-1.0).
            Defaults to 0.65 (recommended by Google).
        cache_dir: Directory for caching downloaded CSV files.
            Defaults to ~/.pygeovision/cache/google_buildings/.
        num_workers: Number of parallel workers for tile processing.
        skip_existing: If True, skip tiles that already have labels.

    Example:
        >>> labeler = GoogleBuildingsLabeler(confidence_threshold=0.75)
        >>> results = labeler.label_tiles(tiles, "./labels/google_buildings/")
        >>> print(f"Labeled {sum(r.success for r in results)} tiles")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        cache_dir: Path | None = None,
        num_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name="google_buildings",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        self.config = GoogleBuildingsConfig(
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir or Path.home() / ".pygeovision" / "cache" / "google_buildings",
            max_workers=num_workers,
        )
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tile_index: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'google_buildings'

    @property
    def supported_tasks(self) -> list:
        return ['segmentation', 'detection']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Generate a building segmentation mask for a single tile.

        Args:
            tile_path: Path to the GeoTIFF tile.
            tile_metadata: TileMetadata with bounds, CRS, and shape info.
            output_path: Destination path for the label GeoTIFF.

        Returns:
            LabelingResult with success status and statistics.
        """
        try:
            import pyproj
            import rasterio
            from shapely.geometry import box
            from shapely.ops import transform as shapely_transform
        except ImportError as exc:
            raise LabelingError(
                "google_buildings labeler requires rasterio, shapely, and pyproj. "
                "Install via: pip install rasterio shapely pyproj"
            ) from exc

        try:
            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                bounds = src.bounds
                transform = src.transform

            # Reproject bounds to EPSG:4326 for GOB query
            wgs84 = pyproj.CRS("EPSG:4326")
            project = pyproj.Transformer.from_crs(
                crs, wgs84, always_xy=True
            ).transform
            tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            tile_box_wgs84 = shapely_transform(project, tile_box)
            bbox_wgs84 = tile_box_wgs84.bounds  # (minx, miny, maxx, maxy)

            # Fetch building polygons from GOB
            polygons = self._fetch_buildings(bbox_wgs84)

            # Rasterize polygons onto mask
            mask = self._rasterize_polygons(
                polygons=polygons,
                bbox=bbox_wgs84,
                height=height,
                width=width,
                src_crs=wgs84,
                dst_crs=crs,
                dst_transform=transform,
            )

            # Write label GeoTIFF
            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_mask_geotiff(mask, output_path, meta)

            stats = self._compute_class_distribution(
                mask,
                class_names={
                    self.config.background_value: "background",
                    self.config.building_value: "building",
                },
            )
            building_coverage = float(np.sum(mask == self.config.building_value)) / mask.size

            return LabelingResult(
                tile_id=str(tile_path),
                label_path=output_path,
                source="google_buildings",
                class_distribution=stats,
                metadata={
                    "building_coverage": building_coverage,
                    "num_buildings": len(polygons),
                    "confidence_threshold": self.config.confidence_threshold,
                    "bbox_wgs84": bbox_wgs84,
                },
            )

        except Exception as exc:
            logger.error(
                "GoogleBuildingsLabeler failed for %s: %s", tile_path, exc
            )
            return LabelingResult(
                tile_id=str(tile_path),
                label_path=None,
                source="google_buildings",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_buildings(
        self, bbox_wgs84: tuple[float, float, float, float]
    ) -> list[dict[str, Any]]:
        """Fetch building polygons from Google Open Buildings for a bounding box.

        Args:
            bbox_wgs84: (minx, miny, maxx, maxy) in WGS-84.

        Returns:
            List of building dicts with 'geometry' and 'confidence' keys.
        """
        from pygeovision.data import footprints as fp  # noqa: PLC0415

        gdf = fp.google_footprints(bbox_wgs84, min_confidence=self.config.confidence_threshold,
                                   cache_dir=self.config.cache_dir)
        buildings = [{"geometry": g, "confidence": float(c)} for g, c in zip(gdf.geometry, gdf["confidence"])]
        logger.debug("Fetched %d buildings (confidence >= %.2f) for bbox %s", len(buildings),
                     self.config.confidence_threshold, bbox_wgs84)
        return buildings

    def _rasterize_polygons(
        self,
        polygons: list[dict[str, Any]],
        bbox: tuple[float, float, float, float],
        height: int,
        width: int,
        src_crs: Any,
        dst_crs: Any,
        dst_transform: Any,
    ) -> np.ndarray:
        """Rasterize building polygons onto a pixel mask.

        Args:
            polygons: List of building dicts with 'geometry' key.
            bbox: Bounding box in src_crs (minx, miny, maxx, maxy).
            height: Output mask height in pixels.
            width: Output mask width in pixels.
            src_crs: Source CRS (WGS-84) for the polygons.
            dst_crs: Destination CRS for the raster tile.
            dst_transform: Affine transform for the raster tile.

        Returns:
            Uint8 numpy array with building=1, background=0.
        """
        mask = np.full((height, width), self.config.background_value, dtype=np.uint8)

        if not polygons:
            return mask

        try:
            import pyproj
            from rasterio.features import rasterize as rio_rasterize
            from shapely.ops import transform as shapely_transform

            # Reproject polygons from WGS-84 to tile CRS
            project = pyproj.Transformer.from_crs(
                src_crs, dst_crs, always_xy=True
            ).transform

            shapes = []
            for bldg in polygons:
                try:
                    reprojected = shapely_transform(project, bldg["geometry"])
                    shapes.append((reprojected.__geo_interface__, self.config.building_value))
                except Exception:
                    continue

            if shapes:
                burned = rio_rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=dst_transform,
                    fill=self.config.background_value,
                    dtype=np.uint8,
                    all_touched=False,
                )
                mask = burned

        except Exception as exc:
            logger.warning("Rasterization failed, returning empty mask: %s", exc)

        return mask
