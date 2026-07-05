"""
pygeovision.data.validators.georeference
=========================================
Validates and, where possible, repairs the georeference of raster files
that have been through an external reproject step (e.g. PyGeoFetch
post_process=["reproject:EPSG:32637", "cog"]).

The specific bug this module addresses
---------------------------------------
After PyGeoFetch's reproject step, some Sentinel-1 GRD output files
arrive with an **identity affine transform** — the CRS field is correctly
set to EPSG:32637 (or whatever the target was) but the actual pixel
coordinates in the file start at (0, 0) instead of real-world UTM metres.

Visible symptoms::

    SAR PRE VH bounds: BoundingBox(left=0.0, bottom=0.0, right=26083.0, top=16700.0)
    Transform: | 1.00, 0.00, 0.00 |
               | 0.00,-1.00, 16700.00 |

The ``a`` component of the affine should be the pixel width in metres
(~10 for Sentinel-1 IW at 10 m resolution, not 1.0). The clip step then
fails with "Input shapes do not overlap raster" because the raster's
extent sits in [0, 26083] × [0, 16700] pixel-space, nowhere near the
actual WGS84 study area bbox.

The validator detects this condition and either:
  * raises ``GeoreferenceCorruptError`` so the calling pipeline can
    decide to re-reproject from the raw download, OR
  * repairs it by running the reprojection itself from the raw file
    (if ``raw_path`` is supplied and the raw file has a valid transform).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

logger = logging.getLogger("pygeovision.validators.georeference")


# ── Exceptions ────────────────────────────────────────────────────────────────

class GeoreferenceCorruptError(ValueError):
    """Raised when a reprojected file has an identity/pixel-space transform
    rather than real-world coordinates."""


class GeoreferenceWarning(UserWarning):
    """Raised when a georeference is suspicious but not definitively corrupt."""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GeoreferenceResult:
    path: str
    valid: bool
    repaired: bool = False
    repaired_path: Optional[str] = None

    # Diagnostics
    srid: Optional[int] = None
    pixel_width_m: Optional[float] = None   # |a| component of affine
    origin_x: Optional[float] = None        # c component
    origin_y: Optional[float] = None        # f component
    bounds: Optional[Tuple] = None

    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# ── Core validator ────────────────────────────────────────────────────────────

def validate_georeference(
    path: str | Path,
    *,
    raw_path: str | Path | None = None,
    repair: bool = True,
    min_pixel_width_m: float = 1.0,
    min_origin_abs: float = 100.0,
) -> GeoreferenceResult:
    """Validate that a raster has a real-world (non-identity) affine transform.

    Parameters
    ----------
    path : str | Path
        Path to the reprojected raster to validate.
    raw_path : str | Path | None
        Path to the original (pre-reproject) download.  If provided and
        ``repair=True``, a corrupt ``path`` is fixed by re-reprojecting
        from ``raw_path``.
    repair : bool
        Whether to attempt repair when corruption is detected.
    min_pixel_width_m : float
        The smallest plausible pixel width in metres.  For Sentinel-1 IW
        at 10 m resolution, the |a| component should be ≈10.  The default
        (1.0) catches the identity transform case (a=1.0) that arises from
        the known PyGeoFetch bug.
    min_origin_abs : float
        The smallest plausible absolute value for the top-left corner
        coordinate.  For UTM projections the origin is on the order of
        hundreds of thousands of metres.  An origin near 0 or 16 700
        (pixel-space height) is a strong indicator of a corrupt transform.

    Returns
    -------
    GeoreferenceResult
        ``.valid`` is True when the transform looks real-world.
        ``.repaired`` is True when a new file was written to ``.repaired_path``.
    """
    path = Path(path)
    result = GeoreferenceResult(path=str(path), valid=False)

    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result

    try:
        with rasterio.open(str(path)) as src:
            t = src.transform
            crs = src.crs
            result.srid = crs.to_epsg() if crs else None
            result.pixel_width_m = abs(t.a)
            result.origin_x = t.c
            result.origin_y = t.f
            result.bounds = src.bounds
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Cannot open raster: {exc}")
        return result

    # ── Diagnostic checks ────────────────────────────────────────────────────

    # Check 1: pixel width — identity transform has a=1.0
    pixel_width_corrupt = result.pixel_width_m <= min_pixel_width_m

    # Check 2: origin near pixel-space zero — corrupt files start at (0,0)
    # or at (0, height_in_pixels), whereas real UTM origins are 100 k–900 k
    origin_near_zero = (
        abs(result.origin_x) < min_origin_abs
        and abs(result.origin_y) < min_origin_abs
    )
    origin_is_pixel_height = (
        abs(result.origin_x) < min_origin_abs
        and abs(result.origin_y) < min_origin_abs * 200  # e.g. 16700
        and abs(result.origin_y) > 0
    )

    corrupt = pixel_width_corrupt or origin_near_zero

    if corrupt:
        result.errors.append(
            f"Identity/pixel-space transform detected: "
            f"a={t.a:.4f} (expected ≥{min_pixel_width_m} m), "
            f"origin=({result.origin_x:.1f}, {result.origin_y:.1f}) "
            f"(expected absolute values ≥{min_origin_abs}). "
            f"This is the known PyGeoFetch post-reproject CRS corruption bug."
        )
        logger.warning("Corrupt georeference in %s: %s", path.name, result.errors[-1])

        if repair and raw_path is not None:
            repaired_path = _repair_from_raw(path, Path(raw_path))
            if repaired_path is not None:
                result.repaired = True
                result.repaired_path = str(repaired_path)
                result.valid = True
                logger.info("Repaired georeference: %s → %s", path.name, repaired_path.name)
            else:
                result.errors.append("Repair failed: could not re-reproject from raw file.")
        return result

    # Soft warnings
    if result.srid is None:
        result.warnings.append("CRS could not be resolved to an EPSG code.")

    result.valid = True
    return result


def assert_valid_georeference(path: str | Path, **kwargs) -> GeoreferenceResult:
    """Like ``validate_georeference`` but raises ``GeoreferenceCorruptError``
    on failure — suitable for use in an assert-like guard at the start of
    any processing step that reads a reprojected file."""
    result = validate_georeference(path, **kwargs)
    if not result.valid:
        raise GeoreferenceCorruptError(
            f"Georeference validation failed for {path}:\n"
            + "\n".join(f"  {e}" for e in result.errors)
        )
    return result


# ── Repair helper ─────────────────────────────────────────────────────────────

def _repair_from_raw(corrupt_path: Path, raw_path: Path) -> Path | None:
    """Re-reproject ``raw_path`` to match the CRS of ``corrupt_path``, writing
    the result adjacent to the corrupt file with a ``_repaired`` suffix."""
    if not raw_path.exists():
        logger.error("Repair skipped: raw_path not found: %s", raw_path)
        return None

    try:
        with rasterio.open(str(corrupt_path)) as corrupt_src:
            target_crs = corrupt_src.crs

        with rasterio.open(str(raw_path)) as raw_src:
            if raw_src.crs is None:
                logger.error("Repair skipped: raw file has no CRS: %s", raw_path)
                return None

            transform, width, height = calculate_default_transform(
                raw_src.crs, target_crs,
                raw_src.width, raw_src.height,
                *raw_src.bounds,
            )
            profile = raw_src.meta.copy()
            profile.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "driver": "GTiff",
                "compress": "lzw",
            })

            repaired_path = corrupt_path.parent / (
                corrupt_path.stem + "_repaired" + corrupt_path.suffix
            )
            with rasterio.open(str(repaired_path), "w", **profile) as dst:
                for band_idx in range(1, raw_src.count + 1):
                    reproject(
                        source=rasterio.band(raw_src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=raw_src.transform,
                        src_crs=raw_src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
        return repaired_path

    except Exception as exc:  # noqa: BLE001
        logger.error("Georeference repair failed: %s", exc)
        return None


# ── Download completeness checker ─────────────────────────────────────────────

def check_download_complete(
    path: str | Path,
    *,
    test_tiles: int = 4,
    test_tile_size: int = 256,
) -> dict:
    """Verify that a downloaded raster file is complete and readable by
    attempting to read several non-overlapping tiles from it.

    This catches the specific failure mode described in the bug report::

        WARNING: Failed to download asset 'vh': The read operation timed out
        ERROR: TIFFFillTile: Read error at row 21504, col 16384

    where a partial download leaves a file on disk that passes an
    ``os.path.exists()`` check but fails when rasterio tries to read a
    tile from the truncated region.

    Parameters
    ----------
    path : str | Path
        Path to check.
    test_tiles : int
        Number of evenly-spaced tiles to test-read.
    test_tile_size : int
        Pixel size (square) of each test tile.

    Returns
    -------
    dict with keys:
        complete (bool), readable_tiles (int), total_tiles (int),
        errors (list[str])
    """
    from rasterio.windows import Window

    path = Path(path)
    result: dict = {"complete": False, "readable_tiles": 0, "total_tiles": test_tiles, "errors": []}

    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    try:
        with rasterio.open(str(path)) as src:
            h, w = src.height, src.width
            # Evenly-spaced tile grid
            row_step = max(1, (h - test_tile_size) // max(1, test_tiles - 1))
            col_step = max(1, (w - test_tile_size) // max(1, test_tiles - 1))

            ok = 0
            for i in range(test_tiles):
                row_off = min(i * row_step, h - test_tile_size)
                col_off = min(i * col_step, w - test_tile_size)
                win = Window(col_off, row_off, test_tile_size, test_tile_size)
                try:
                    data = src.read(1, window=win)
                    if data.shape == (test_tile_size, test_tile_size):
                        ok += 1
                    else:
                        result["errors"].append(
                            f"Tile {i}: unexpected shape {data.shape}"
                        )
                except Exception as exc:  # noqa: BLE001
                    result["errors"].append(f"Tile {i} at row={row_off},col={col_off}: {exc}")

            result["readable_tiles"] = ok
            result["complete"] = ok == test_tiles

    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Cannot open file: {exc}")

    return result


# ── Bbox reprojection helper (used by SAR pipeline) ──────────────────────────

def reproject_bbox_to_raster_crs(
    bbox_wgs84: tuple,
    raster_path: str | Path,
) -> tuple:
    """Transform a WGS84 (lon_min, lat_min, lon_max, lat_max) bbox to the
    native CRS of a raster — the correct way to prepare a clip bbox when
    the raster is in a projected CRS (e.g. UTM).

    This is the fix for the "Input shapes do not overlap raster" clip error
    that occurs when the study area bbox is in WGS84 degrees but the raster
    is in UTM metres — they share no coordinates and rasterio's clip cannot
    find any overlap.

    Returns the bbox in the raster's CRS as (left, bottom, right, top).
    """
    from pyproj import Transformer

    with rasterio.open(str(raster_path)) as src:
        raster_crs = src.crs

    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    raster_epsg = raster_crs.to_epsg()
    if raster_epsg == 4326:
        return bbox_wgs84  # already WGS84, no transform needed

    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    left, bottom = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    right, top = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (left, bottom, right, top)
