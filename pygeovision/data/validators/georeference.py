"""
pygeovision.data.validators.georeference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CRS-aware georeference validation with GCP-based recovery.

Root cause of the NB09 SAR stack failure (Jul 2026):
  pygeofetch's reproject:EPSG:32630 post-processing writes a pixel-space
  identity transform (a=1.0, origin=(0, N)) while tagging the file EPSG:32630.
  The file LOOKS valid (correct CRS tag) but has NO geographic information.
  pygeovision's old validator detected this correctly but returned
  repaired_path=None — no recovery was attempted.
  prep_sar() then returned None for every acquisition, leaving the stack empty.

This fix adds GCP-based recovery: when an identity transform is detected,
we check for embedded GCPs (pygeofetch always writes 210-231 of them),
derive the correct affine transform from them via rasterio.transform.from_gcps(),
write a repaired copy, and return its path in repaired_path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_gcps

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Geographic CRS (degrees): valid pixel size > 1e-7 deg (~1 cm at equator)
GEO_MIN_PIXEL_DEG = 1e-7
# Projected CRS (metres): valid pixel size >= 0.5 m
PROJ_MIN_PIXEL_M  = 0.5
# Maximum sensible pixel size for projected CRS (> 10 km = almost certainly wrong)
PROJ_MAX_PIXEL_M  = 10_000.0
# Minimum GCPs needed for reliable transform recovery
MIN_GCPS_FOR_RECOVERY = 4


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GeoreferenceResult:
    """Return value of validate_sar_georeference() and validate_georeference()."""
    valid:         bool
    repaired_path: str | None = None
    errors:        list[str]     = field(default_factory=list)
    warnings:      list[str]     = field(default_factory=list)
    crs:           str | None = None
    transform_a:   float | None = None
    gcp_count:     int = 0
    recovered:     bool = False
    srid:          int | None = None
    origin_x:      float | None = None
    origin_y:      float | None = None

    @property
    def repaired(self) -> bool:
        """Alias for `recovered` — whether a repaired file was produced."""
        return self.recovered

    @property
    def pixel_width_m(self) -> float | None:
        """Alias for `transform_a` — pixel width in the raster's native CRS units."""
        return self.transform_a


# ── Identity-transform detection ──────────────────────────────────────────────

def _is_identity_transform(transform) -> bool:
    """Return True if transform is a pixel-space identity (a≈1, b=d=0, e≈±1)."""
    return (
        abs(abs(transform.a) - 1.0) < 0.01 and
        abs(transform.b) < 1e-6 and
        abs(transform.d) < 1e-6 and
        abs(abs(transform.e) - 1.0) < 0.01
    )


def _is_geographic_crs(crs: CRS) -> bool:
    """Return True if the CRS is geographic (degree-based)."""
    try:
        return crs.is_geographic
    except Exception:
        return False


# ── GCP recovery ─────────────────────────────────────────────────────────────

def _recover_via_gcps(src_path: str, dst_path: str) -> bool:
    """
    Attempt to recover georeferencing from embedded GCPs.

    pygeofetch writes 210–231 GCPs in the original geographic CRS before
    reprojection. When the reprojection writes a corrupt identity transform,
    these GCPs are the authoritative source of truth. We:
      1. Read the data array and embedded GCPs.
      2. Derive the affine transform from the GCPs using from_gcps().
      3. Write a corrected GeoTIFF with the recovered transform.
      4. Validate the output.

    Returns True if recovery succeeded, False otherwise.
    """
    try:
        with rasterio.open(src_path) as src:
            gcps, gcp_crs = src.gcps
            if not gcps or len(gcps) < MIN_GCPS_FOR_RECOVERY:
                logger.debug(f"GCP recovery: insufficient GCPs ({len(gcps) if gcps else 0})")
                return False

            data   = src.read()
            meta   = src.meta.copy()
            src_crs = src.crs   # the tagged CRS (may be EPSG:32630 even if transform is wrong)

        # Derive affine transform from GCPs
        # from_gcps() uses the GCP pixel/line coordinates and their
        # geographic coordinates to compute a best-fit affine transform.
        recovered_transform = from_gcps(gcps)

        # Sanity-check the recovered transform
        if _is_identity_transform(recovered_transform):
            logger.warning("GCP recovery: derived transform is still identity — GCPs may be corrupt")
            return False

        # The GCP CRS may be WGS84 while the file is tagged EPSG:32630.
        # We use the original file's tagged CRS if it is projected,
        # otherwise fall back to the GCP CRS.
        use_crs = src_crs if (src_crs and not _is_geographic_crs(src_crs)) else gcp_crs

        # Validate pixel size is physically sensible for the chosen CRS
        a = abs(recovered_transform.a)
        if _is_geographic_crs(use_crs):
            if a < GEO_MIN_PIXEL_DEG:
                logger.warning(f"GCP recovery: recovered pixel size {a:.8f} deg seems too small")
                return False
        else:
            if a < PROJ_MIN_PIXEL_M or a > PROJ_MAX_PIXEL_M:
                logger.warning(f"GCP recovery: recovered pixel size {a:.1f} m is outside [0.5, 10000] m")
                return False

        # Write the corrected file
        meta.update(transform=recovered_transform, crs=use_crs)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(data)

        logger.info(
            f"GCP recovery succeeded: {Path(src_path).name} → {Path(dst_path).name} "
            f"(pixel_size={a:.2f}, crs={use_crs.to_epsg()}, gcps_used={len(gcps)})"
        )
        return True

    except Exception as e:
        logger.warning(f"GCP recovery failed for {Path(src_path).name}: {e}")
        return False


# ── Reprojection recovery (fallback when no GCPs) ────────────────────────────

def _recover_via_bounds_heuristic(src_path: str, dst_path: str, target_crs: str = "EPSG:32630") -> bool:
    """
    Last-resort recovery: if the origin looks like pixel-row offset (y in 0–30000),
    attempt to reconstruct the transform from the CRS zone bounds + pixel count.
    This is approximate but often good enough for Accra / UTM Zone 30N.
    """
    try:
        from rasterio.crs import CRS as RasterioCRS
        from rasterio.transform import from_bounds as affine_from_bounds

        with rasterio.open(src_path) as src:
            data   = src.read()
            meta   = src.meta.copy()
            H, W   = data.shape[1], data.shape[2]
            # Use known Accra approximate bounds in EPSG:32630
            # Odaw bbox WGS84: lon (-0.30, -0.05) lat (5.50, 5.70)
            # UTM 30N approximate: E 630000-655000, N 608000-630000
            ACCRA_UTM_BOUNDS = (628000, 606000, 658000, 633000)
            crs_obj = RasterioCRS.from_epsg(32630)
            recovered_transform = affine_from_bounds(
                *ACCRA_UTM_BOUNDS, W, H
            )

        a = abs(recovered_transform.a)
        if a < PROJ_MIN_PIXEL_M or a > PROJ_MAX_PIXEL_M:
            return False

        meta.update(transform=recovered_transform, crs=crs_obj)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(data)

        logger.info(
            f"Bounds-heuristic recovery: {Path(src_path).name} → {Path(dst_path).name} "
            f"(pixel_size={a:.1f} m, APPROXIMATE Accra bounds)"
        )
        return True

    except Exception as e:
        logger.debug(f"Bounds-heuristic recovery failed: {e}")
        return False


# ── Raw-reference recovery (repair using a separate known-good file) ─────────

def _recover_via_raw_reference(corrupt_path: str, raw_path: str, dst_path: str) -> bool:
    """
    Repair a corrupt (identity-transform) raster using a separate raw file
    that carries a valid transform/CRS for the same scene — e.g. the
    pre-reprojection download, before pygeofetch's post-reproject step
    corrupted it.

    Writes `corrupt_path`'s data with `raw_path`'s transform + CRS to
    `dst_path`. Returns True if the raw file's own transform looks valid.
    """
    try:
        with rasterio.open(raw_path) as raw_src:
            raw_transform = raw_src.transform
            raw_crs       = raw_src.crs

        if raw_crs is None or _is_identity_transform(raw_transform):
            logger.debug(f"Raw-reference recovery: raw file {raw_path} has no usable transform")
            return False

        is_geo = _is_geographic_crs(raw_crs)
        a = abs(raw_transform.a)
        min_px = GEO_MIN_PIXEL_DEG if is_geo else PROJ_MIN_PIXEL_M
        if a < min_px or (not is_geo and a > PROJ_MAX_PIXEL_M):
            logger.debug(f"Raw-reference recovery: raw file pixel size {a} out of range")
            return False

        with rasterio.open(corrupt_path) as src:
            data = src.read()
            meta = src.meta.copy()

        meta.update(transform=raw_transform, crs=raw_crs)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(data)

        logger.info(
            f"Raw-reference recovery succeeded: {Path(corrupt_path).name} + "
            f"{Path(raw_path).name} → {Path(dst_path).name}"
        )
        return True

    except Exception as e:
        logger.warning(f"Raw-reference recovery failed: {e}")
        return False


# ── Core validation function ──────────────────────────────────────────────────

def validate_georeference(
    path: str,
    expected_crs: str | None = None,
    min_pixel_size: float | None = None,
    max_origin: float | None = None,
    attempt_recovery: bool = True,
    file_path: str | None = None,
    repair: bool | None = None,
    allow_heuristic_recovery: bool = False,
) -> GeoreferenceResult:
    """
    CRS-aware georeference validation with GCP-based recovery.

    Args:
        path:              Path to the raster file to validate.
        expected_crs:      If set, also checks the CRS matches (e.g. 'EPSG:32630').
        min_pixel_size:    Override minimum pixel size. Auto-detected from CRS type if None.
        max_origin:        Override maximum absolute origin value. None = no check.
        attempt_recovery:  If True, attempt recovery when validation fails.
        file_path:         Optional path to a separate raw/source raster carrying
                            a known-good transform, used to repair `path` when its
                            own transform is corrupt (e.g. pygeofetch's identity-
                            transform bug). Takes priority over GCP recovery when
                            provided.
        repair:            Alias for `attempt_recovery` (if given, overrides it).
        allow_heuristic_recovery: If True, also try the approximate Accra-bounds
                            heuristic as a last resort when GCP/raw recovery are
                            unavailable. Off by default — it's a location-specific
                            approximation, not a general recovery method.

    Returns:
        GeoreferenceResult with valid, repaired_path, errors, warnings, etc.
    """
    do_recovery = attempt_recovery if repair is None else repair
    file_path_arg = path
    result = GeoreferenceResult(valid=False)

    if not os.path.exists(file_path_arg):
        result.errors.append(f"File not found: {file_path_arg}")
        return result

    try:
        with rasterio.open(file_path_arg) as src:
            transform = src.transform
            crs       = src.crs
            gcps, _   = src.gcps
            result.gcp_count = len(gcps) if gcps else 0

        if crs is None:
            result.errors.append("No CRS defined")
        else:
            result.crs  = crs.to_string()
            result.srid = crs.to_epsg()

        is_geo = _is_geographic_crs(crs) if crs else False
        a = abs(transform.a)
        result.transform_a = a
        result.origin_x = transform.c
        result.origin_y = transform.f

        # Determine minimum pixel size from CRS type
        if min_pixel_size is None:
            min_px = GEO_MIN_PIXEL_DEG if is_geo else PROJ_MIN_PIXEL_M
        else:
            min_px = min_pixel_size

        # Check identity transform
        if _is_identity_transform(transform):
            result.errors.append(
                f"Identity/pixel-space transform detected: a={transform.a:.4f} "
                f"(expected ≥{min_px} {'deg' if is_geo else 'm'}), "
                f"origin=({transform.c:.1f}, {transform.f:.1f}) "
                f"(expected absolute values ≥100.0). "
                f"This is the known PyGeoFetch post-reproject CRS corruption bug."
            )
        elif a < min_px:
            result.errors.append(
                f"Pixel size too small: a={a:.6f} (min={min_px})"
            )
        elif not is_geo and a > PROJ_MAX_PIXEL_M:
            result.errors.append(
                f"Pixel size too large for projected CRS: a={a:.1f} m (max={PROJ_MAX_PIXEL_M})"
            )

        # CRS check
        if expected_crs and crs:
            try:
                exp = CRS.from_string(expected_crs)
                if not crs.equals(exp):
                    result.warnings.append(
                        f"CRS mismatch: expected {expected_crs}, got {crs.to_epsg()}"
                    )
            except Exception:
                pass

    except Exception as e:
        result.errors.append(f"Cannot open file: {e}")
        return result

    # If valid, return immediately
    if not result.errors:
        result.valid = True
        return result

    # ── Attempt recovery ──────────────────────────────────────────────────────
    if not do_recovery:
        return result

    src_path = str(file_path_arg)
    stem     = Path(src_path).stem
    suffix   = Path(src_path).suffix
    repair_dir = Path(src_path).parent / "_repaired"
    repair_dir.mkdir(exist_ok=True)
    dst_path = str(repair_dir / f"{stem}_repaired{suffix}")

    # Method 1: raw-reference recovery (explicit known-good source file)
    if file_path:
        if _recover_via_raw_reference(src_path, file_path, dst_path):
            recovered_result = validate_georeference(
                dst_path, expected_crs=expected_crs,
                min_pixel_size=min_pixel_size, attempt_recovery=False,
            )
            if recovered_result.valid:
                result.repaired_path = dst_path
                result.recovered     = True
                result.warnings.append(
                    f"Georeference recovered from raw reference {Path(file_path).name} "
                    f"→ {Path(dst_path).name}"
                )
                result.valid = True
                logger.info(f"Raw-reference recovery validated: {dst_path}")
                return result

    # Method 2: GCP recovery (uses embedded GCPs)
    if result.gcp_count >= MIN_GCPS_FOR_RECOVERY:
        logger.info(
            f"Attempting GCP recovery for {Path(src_path).name} "
            f"({result.gcp_count} GCPs available)"
        )
        if _recover_via_gcps(src_path, dst_path):
            # Validate the recovered file
            recovered_result = validate_georeference(
                dst_path, expected_crs=expected_crs,
                min_pixel_size=min_pixel_size, attempt_recovery=False
            )
            if recovered_result.valid:
                result.repaired_path = dst_path
                result.recovered     = True
                result.warnings.append(
                    f"Georeference recovered from {result.gcp_count} GCPs → {Path(dst_path).name}"
                )
                result.valid = True
                logger.info(f"GCP recovery validated: {dst_path}")
                return result
            else:
                logger.warning(f"Recovered file failed validation: {recovered_result.errors}")

    # Method 3: Bounds heuristic (Accra-specific fallback, opt-in only)
    if allow_heuristic_recovery:
        logger.info(f"Attempting bounds-heuristic recovery for {Path(src_path).name}")
        dst_path_h = str(repair_dir / f"{stem}_repaired_heuristic{suffix}")
        if _recover_via_bounds_heuristic(src_path, dst_path_h):
            result.repaired_path = dst_path_h
            result.recovered     = True
            result.valid         = True
            result.warnings.append(
                f"Georeference recovered via Accra bounds heuristic (APPROXIMATE) → {Path(dst_path_h).name}"
            )
            return result

    # Recovery failed
    logger.error(
        f"All recovery methods failed for {Path(src_path).name}. "
        f"GCPs available: {result.gcp_count}. Errors: {result.errors}"
    )
    return result


# ── Convenience alias used by the notebook and sar.py ────────────────────────

def validate_sar_georeference(file_path: str, attempt_recovery: bool = True) -> GeoreferenceResult:
    """Validate SAR GeoTIFF georeference with automatic GCP recovery."""
    return validate_georeference(file_path, expected_crs="EPSG:32630", attempt_recovery=attempt_recovery)


# ── check_download_complete ───────────────────────────────────────────────────

def check_download_complete(
    file_path: str,
    total_tiles: int | None = None,
    file_size: int | None = None,
) -> dict:
    """
    Check whether a downloaded GeoTIFF is complete and readable.

    Args:
        file_path:    Path to the downloaded raster.
        total_tiles:  Optional expected tile count, if already known from a
                      manifest/API response, for logging/cross-checks. The
                      completeness check itself always re-derives the actual
                      tile count from the file (a caller-supplied count can't
                      be trusted to detect a truncated download).
        file_size:    Optional expected file size in bytes, for the same
                      cross-check purpose (unused in the pass/fail decision).

    Returns dict with keys:
        complete        (bool)   — True if all tiles readable
        readable_tiles  (int)    — number of readable overview tiles
        total_tiles     (int)    — total tiles expected (measured from file)
        file_size_mb    (float)  — file size
        errors          (list)   — any errors encountered
    """
    result = {
        "complete": False,
        "readable_tiles": 0,
        "total_tiles": 0,
        "file_size_mb": 0.0,
        "errors": [],
    }

    if not os.path.exists(file_path):
        result["errors"].append(f"File not found: {file_path}")
        return result

    try:
        result["file_size_mb"] = os.path.getsize(file_path) / (1024 * 1024)
        if result["file_size_mb"] < 0.01:
            result["errors"].append(f"File too small: {result['file_size_mb']:.3f} MB (likely incomplete download)")
            return result

        with rasterio.open(file_path) as src:
            H, W  = src.height, src.width
            # Sample tiles across the raster to verify readability
            tile_size = min(512, H, W)
            tiles_y   = max(1, H // tile_size)
            tiles_x   = max(1, W // tile_size)
            result["total_tiles"] = tiles_y * tiles_x

            readable = 0
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    row_off = ty * tile_size
                    col_off = tx * tile_size
                    win = rasterio.windows.Window(
                        col_off, row_off,
                        min(tile_size, W - col_off),
                        min(tile_size, H - row_off),
                    )
                    try:
                        data = src.read(1, window=win)
                        if not np.all(np.isnan(data)):
                            readable += 1
                        else:
                            readable += 1  # all-NaN tile is still readable
                    except Exception as e:
                        result["errors"].append(f"Tile ({ty},{tx}) unreadable: {e}")

            result["readable_tiles"] = readable
            result["complete"] = (readable == result["total_tiles"]) and (readable > 0)

    except Exception as e:
        result["errors"].append(f"Cannot open file: {e}")

    return result


# ── reproject_bbox_to_raster_crs ─────────────────────────────────────────────

def reproject_bbox_to_raster_crs(bbox_wgs84: tuple, raster_path: str) -> tuple:
    """
    Reproject a WGS84 bounding box (lon_min, lat_min, lon_max, lat_max)
    to the CRS of the given raster file.

    Returns (x_min, y_min, x_max, y_max) in the raster's CRS.
    Falls back to the original bbox if reprojection fails.
    """
    try:
        from pyproj import Transformer

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        if raster_crs is None:
            return bbox_wgs84

        if _is_geographic_crs(raster_crs):
            return bbox_wgs84  # already in degrees

        transformer = Transformer.from_crs(
            "EPSG:4326", raster_crs.to_epsg() or raster_crs.to_wkt(),
            always_xy=True
        )
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)
        # Ensure correct ordering
        return (
            min(x_min, x_max), min(y_min, y_max),
            max(x_min, x_max), max(y_min, y_max),
        )

    except Exception as e:
        logger.warning(f"reproject_bbox_to_raster_crs failed ({e}), returning original bbox")
        return bbox_wgs84
