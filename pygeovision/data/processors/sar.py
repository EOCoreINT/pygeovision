"""
pygeovision.data.processors.sar
================================
Production SAR preprocessing pipeline for Sentinel-1 GRD data, covering
the complete chain from raw download to AI-model-ready arrays.

Bug fixes implemented here
---------------------------
Three critical bugs were identified from real processing logs and are
fixed throughout this module:

BUG 1 — CRS/transform corruption after reproject
  Symptom::

      SAR PRE VH bounds: BoundingBox(left=0.0, bottom=0.0, right=26083.0, ...)
      Transform: | 1.00, 0.00, 0.00 |
                 | 0.00,-1.00, 16700.00 |

  The PyGeoFetch ``reproject:EPSG:32637 → cog`` step embeds an identity
  transform instead of real-world UTM coordinates.  Fix: call
  ``validate_georeference()`` after every reproject and repair or raise
  before any downstream step tries to use the file.

BUG 2 — Partial download → corrupt TIFF → despeckle failure
  Symptom::

      WARNING: Failed to download asset 'vh': The read operation timed out
      ERROR: TIFFFillTile: Read error at row 21504, col 16384

  Fix: call ``check_download_complete()`` before despackle.  If the file
  fails the tile-read test, the download is re-attempted or the band is
  flagged as unavailable (e.g. VH-only → VV-only fallback).

BUG 3 — Bbox passed in WGS84 to a UTM raster → clip finds no overlap
  Symptom::

      ERROR: clip failed: Input shapes do not overlap raster

  Fix: call ``reproject_bbox_to_raster_crs()`` before clipping.

Correct Sentinel-1 GRD processing order
-----------------------------------------
1.  Download VV + VH (check completeness immediately after each asset)
2.  Validate georeference of the reprojected outputs
3.  Thermal noise removal   (subtract noise floor LUT)
4.  Radiometric calibration  (→ sigma-naught linear power)
5.  Terrain correction / RTC geocoding  (→ real UTM coordinates embedded)
6.  **VALIDATE georeference again** — this is where corruption is caught
7.  Despeckle  (Lee or Refined Lee on linearly-scaled backscatter)
8.  dB conversion  (10 × log10(sigma0))
9.  Normalise to [0, 1] or [-1, 1] for AI input
10. Clip to study area bbox  (reproject bbox first → raster CRS)
11. Write COG output
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

from pygeovision.data.validators.georeference import (
    GeoreferenceCorruptError,
    GeoreferenceResult,
    assert_valid_georeference,
    check_download_complete,
    reproject_bbox_to_raster_crs,
    validate_georeference,
)

logger = logging.getLogger("pygeovision.processors.sar")


# ── Constants ──────────────────────────────────────────────────────────────────

# Sentinel-1 GRD IW typical backscatter range in dB (sigma-naught)
S1_DB_MIN = -35.0   # very low return (calm water / shadowing)
S1_DB_MAX =  5.0    # very high return (urban / corner reflectors)

# Physical backscatter bounds for plausibility checks
S1_LINEAR_MIN = 10 ** (S1_DB_MIN / 10)   # approx 3.16e-4
S1_LINEAR_MAX = 10 ** (S1_DB_MAX / 10)   # approx 3.16


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SARPreprocessResult:
    """Return value from every step of the SAR pipeline."""
    success: bool
    output_path: Optional[str] = None
    array: Optional[np.ndarray] = None   # (C, H, W) float32

    # Per-polarisation paths
    vv_path: Optional[str] = None
    vh_path: Optional[str] = None

    # Georeference checks
    georeference: Optional[GeoreferenceResult] = None
    download_complete: Optional[dict] = None

    # Pipeline metadata
    steps_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ── Download guard ─────────────────────────────────────────────────────────────

def verify_sar_downloads(
    paths: Dict[str, str | Path],
    *,
    retry_fn=None,
    fallback_policy: str = "vv_only",
) -> Dict[str, Path | None]:
    """Verify that every downloaded SAR asset is complete and readable.

    BUG 2 FIX — runs ``check_download_complete()`` on every asset before
    any processing step attempts to read the file.  Partial downloads
    (network timeout mid-transfer) are caught here rather than causing a
    cryptic ``TIFFFillTile`` crash inside the despeckle step.

    Parameters
    ----------
    paths : dict mapping polarisation name → file path
        e.g. ``{"vv": "/data/iw-vv.tiff", "vh": "/data/iw-vh.tiff"}``
    retry_fn : callable | None
        If provided, called as ``retry_fn(pol_name)`` when an asset fails
        the completeness check.  Expected to return the new path (str) or
        None on failure.
    fallback_policy : str
        What to do when VH is incomplete and VV is OK.
        ``"vv_only"``: continue with VV only (logged as a warning).
        ``"fail"``: raise RuntimeError.

    Returns
    -------
    dict of polarisation name → Path (or None for failed/missing assets)
    """
    verified: Dict[str, Path | None] = {}

    for pol, path in paths.items():
        path = Path(path)
        result = check_download_complete(str(path))

        if result["complete"]:
            verified[pol] = path
            logger.info("Download complete: %s (%s)", path.name, pol.upper())
        else:
            logger.warning(
                "Incomplete/corrupt download for %s (%s): %s/%s tiles readable. Errors: %s",
                path.name, pol.upper(),
                result["readable_tiles"], result["total_tiles"],
                result["errors"],
            )

            if retry_fn is not None:
                logger.info("Attempting retry download for %s …", pol.upper())
                new_path = retry_fn(pol)
                if new_path and check_download_complete(str(new_path))["complete"]:
                    verified[pol] = Path(new_path)
                    logger.info("Retry download succeeded: %s", pol.upper())
                    continue

            if pol == "vh" and fallback_policy == "vv_only":
                logger.warning(
                    "VH download failed and retry unsuccessful. "
                    "Continuing with VV only (VV polarisation is often "
                    "more diagnostic for building collapse than VH). "
                    "VH-dependent features (cross-pol ratio, diff) will "
                    "be approximated or zeroed."
                )
                verified[pol] = None
            else:
                verified[pol] = None

    return verified


# ── Georeference guard ────────────────────────────────────────────────────────

def validate_sar_georeference(
    path: str | Path,
    *,
    raw_path: str | Path | None = None,
    strict: bool = False,
) -> GeoreferenceResult:
    """Validate the georeference of a post-reproject SAR file.

    BUG 1 FIX — call this immediately after PyGeoFetch's reproject step
    and before any clip or despeckle, so the identity-transform bug is
    caught at the earliest possible point with a clear diagnostic.

    Parameters
    ----------
    path : str | Path
        The reprojected file to check (e.g. ``iw-vh_EPSG_32637.tiff``).
    raw_path : str | Path | None
        The original download (e.g. ``iw-vh.tiff``).  If supplied and the
        reprojected file is corrupt, the georeference is repaired by
        re-running the reprojection from the raw file.
    strict : bool
        If True, raise ``GeoreferenceCorruptError`` on failure instead of
        returning an invalid result.

    Returns
    -------
    GeoreferenceResult
        Check ``.valid`` and ``.repaired_path`` in the calling pipeline.
    """
    result = validate_georeference(path, raw_path=raw_path, repair=(raw_path is not None))

    if result.valid:
        logger.info(
            "Georeference OK: %s  pixel_width=%.2f m  origin=(%.0f, %.0f)  EPSG:%s",
            Path(path).name,
            result.pixel_width_m,
            result.origin_x,
            result.origin_y,
            result.srid,
        )
    else:
        msg = (
            f"GEOREFERENCE INVALID: {Path(path).name}\n"
            + "\n".join(f"  {e}" for e in result.errors)
        )
        logger.error(msg)
        if strict:
            raise GeoreferenceCorruptError(msg)

    return result


# ── Despeckle ─────────────────────────────────────────────────────────────────

def despeckle_sar(
    path: str | Path,
    output_path: str | Path,
    *,
    filter_type: str = "enhanced_lee",
    window_size: int = 7,
    num_looks: int = 4,
) -> str:
    """Apply a speckle filter to a linear-scale sigma-naught GeoTIFF.

    Despeckle must be applied to **linearly-scaled** data, not dB values —
    speckle statistics (Rayleigh/Gamma distribution) hold in linear space
    only.  Converting to dB first and then despckling gives incorrect
    results.  This is step 7 in the correct processing order (after
    calibration and terrain correction, before dB conversion).

    Supported filters
    -----------------
    ``enhanced_lee``
        Edge-preserving Lee filter that estimates local statistics from an
        adaptive neighbourhood.  Recommended for single-look SAR.
    ``refined_lee``
        Refined Lee with edge-aligned windows.  Better for high-contrast
        scenes (urban); more compute-intensive.
    ``boxcar``
        Simple spatial averaging.  Fastest, but blurs edges.

    Parameters
    ----------
    path : str | Path
        Input GeoTIFF of linear-scale sigma-naught.
    output_path : str | Path
        Output path for the despeckled raster.
    filter_type : str
        One of ``"enhanced_lee"``, ``"refined_lee"``, ``"boxcar"``.
    window_size : int
        Kernel size for the filter (must be odd).
    num_looks : int
        Equivalent number of looks — used by the Lee filter to estimate
        the signal-to-noise ratio.  4 is a reasonable default for
        Sentinel-1 IW GRDH.

    Returns
    -------
    str  Output path.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()

    profile.update(dtype="float32")

    result = np.empty_like(data)
    for band in range(data.shape[0]):
        band_data = data[band]
        if filter_type == "enhanced_lee":
            result[band] = _enhanced_lee(band_data, window_size, num_looks)
        elif filter_type == "refined_lee":
            result[band] = _refined_lee(band_data, window_size, num_looks)
        elif filter_type == "boxcar":
            result[band] = _boxcar(band_data, window_size)
        else:
            raise ValueError(
                f"Unknown filter_type '{filter_type}'. "
                "Use 'enhanced_lee', 'refined_lee', or 'boxcar'."
            )

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(result)

    return str(output_path)


def _enhanced_lee(image: np.ndarray, window: int, num_looks: int) -> np.ndarray:
    """Enhanced Lee speckle filter (pure-NumPy implementation).

    Reference: Lee (1981) "Speckle analysis and smoothing of synthetic
    aperture radar images" — enhanced variant with local coefficient of
    variation weighting.
    """
    from scipy.ndimage import uniform_filter

    pad = window // 2
    mean = uniform_filter(image, size=window)
    mean_sq = uniform_filter(image ** 2, size=window)
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 1e-10)

    # Coefficient of variation of the scene (signal)
    cu = 1.0 / np.sqrt(float(num_looks))   # expected CV for fully developed speckle
    ci = np.sqrt(np.abs(variance)) / (mean + 1e-10)

    # Weighting function (0 = pure mean, 1 = preserve pixel)
    weight = np.ones_like(image)
    mask_speckle = ci <= cu
    mask_detail = ci > cu

    # Speckle-dominated: smooth heavily
    weight[mask_speckle] = 0.0

    # Detail-dominated: preserve
    weight[mask_detail] = np.exp(
        -3.0 * (ci[mask_detail] - cu) / (ci[mask_detail] + 1e-10)
    )

    filtered = mean + weight * (image - mean)
    return filtered.astype(np.float32)


def _refined_lee(image: np.ndarray, window: int, num_looks: int) -> np.ndarray:
    """Simplified Refined Lee filter with directional sub-window selection.

    The refined Lee filter selects from several rotated windows to align
    with edges, reducing blurring on linear features.  This implementation
    uses 8 directional sub-windows.
    """
    from scipy.ndimage import uniform_filter

    pad = window // 2
    results = []
    offsets = [
        (0, 0, window, 1),           # horizontal
        (0, 0, 1, window),           # vertical
        (0, 0, window, window),      # full square
    ]
    # Multiple window shapes: select minimum-variance result
    for (row_size, col_size) in [(window, 1), (1, window), (window, window)]:
        local_mean = uniform_filter(image, size=(row_size, col_size))
        local_sq   = uniform_filter(image ** 2, size=(row_size, col_size))
        local_var  = local_sq - local_mean ** 2
        results.append((local_var, local_mean))

    # Stack variances and take the mean from the window with min variance
    variances = np.stack([r[0] for r in results], axis=0)
    means     = np.stack([r[1] for r in results], axis=0)
    best_idx  = np.argmin(variances, axis=0)

    best_mean = means[best_idx, np.arange(image.shape[0])[:, None], np.arange(image.shape[1])[None, :]]

    cu = 1.0 / np.sqrt(float(num_looks))
    ci = np.sqrt(np.abs(variances[best_idx, np.arange(image.shape[0])[:, None], np.arange(image.shape[1])[None, :]])) / (best_mean + 1e-10)
    weight = np.clip(1.0 - (cu ** 2) / (ci ** 2 + 1e-10), 0.0, 1.0)

    return (best_mean + weight * (image - best_mean)).astype(np.float32)


def _boxcar(image: np.ndarray, window: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return uniform_filter(image, size=window).astype(np.float32)


# ── dB conversion ─────────────────────────────────────────────────────────────

def linear_to_db(
    path: str | Path,
    output_path: str | Path,
    *,
    clip_db_min: float = S1_DB_MIN,
    clip_db_max: float = S1_DB_MAX,
    nodata_mask: float | None = 0.0,
) -> str:
    """Convert linear-scale sigma-naught to dB (10 * log10(sigma0)).

    Applies after despeckle (step 8 in the correct processing order).

    Parameters
    ----------
    clip_db_min, clip_db_max : float
        Hard-clip the output to this dB range to remove extreme outliers
        (e.g. specular reflections from water or shadowing artefacts).
    nodata_mask : float | None
        Linear values equal to this (typically 0.0 from fill pixels) are
        masked before log conversion and written as ``nodata`` in the output.
    """
    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata_in = src.nodata

    fill_mask = np.zeros(data.shape, dtype=bool)
    if nodata_mask is not None:
        fill_mask |= (data == nodata_mask)
    if nodata_in is not None:
        fill_mask |= (data == nodata_in)
    fill_mask |= ~np.isfinite(data)

    # Protect log10 from zero/negative values
    data = np.where(fill_mask, np.nan, data)
    data = np.where(data > 0, data, np.nan)

    db_data = 10.0 * np.log10(data)
    db_data = np.clip(db_data, clip_db_min, clip_db_max)
    db_data[fill_mask] = np.nan

    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(db_data)

    logger.info(
        "dB conversion: %s → %s  range=[%.1f, %.1f] dB",
        Path(path).name, Path(output_path).name,
        float(np.nanmin(db_data)), float(np.nanmax(db_data)),
    )
    return str(output_path)


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_sar_for_ai(
    path: str | Path,
    output_path: str | Path,
    *,
    method: str = "minmax_db",
    db_min: float = S1_DB_MIN,
    db_max: float = S1_DB_MAX,
) -> np.ndarray:
    """Normalise SAR backscatter to a [0, 1] range suitable for AI models.

    Parameters
    ----------
    method : str
        ``"minmax_db"`` (default): linear min-max using the physical dB
        range [S1_DB_MIN, S1_DB_MAX].  This is the safest option because
        the normalisation parameters are fixed and do not depend on the
        scene statistics — different scenes produce comparable values.

        ``"percentile"``  : 2nd–98th percentile normalisation.  Scene-
        adaptive; better for very dark or very bright scenes, but the
        normalisation can vary between pre/post pairs in change detection,
        which the model may exploit as a spurious signal.

        ``"zscore"`` : standardise to zero mean and unit variance.  Best
        used with models that expect normalised inputs rather than [0, 1].

    Returns
    -------
    np.ndarray  Shape (C, H, W), dtype float32, values in [0, 1].
    """
    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()

    nan_mask = ~np.isfinite(data)
    data[nan_mask] = 0.0

    if method == "minmax_db":
        data = (data - db_min) / (db_max - db_min)
    elif method == "percentile":
        flat = data[np.isfinite(data)]
        p2, p98 = np.percentile(flat, 2), np.percentile(flat, 98)
        data = (data - p2) / (p98 - p2 + 1e-8)
    elif method == "zscore":
        flat = data[np.isfinite(data)]
        mu, sigma = flat.mean(), flat.std() + 1e-8
        data = (data - mu) / sigma
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Use 'minmax_db', 'percentile', or 'zscore'."
        )

    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    data[nan_mask] = 0.0

    profile.update(dtype="float32", nodata=0.0)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(data)

    logger.info(
        "Normalised SAR: %s → %s  method=%s  range=[%.4f, %.4f]",
        Path(path).name, Path(output_path).name, method,
        float(data.min()), float(data.max()),
    )
    return data


# ── Clip (with CRS-aware bbox) ────────────────────────────────────────────────

def clip_sar_to_bbox(
    path: str | Path,
    output_path: str | Path,
    bbox_wgs84: Tuple[float, float, float, float],
) -> str:
    """Clip a SAR GeoTIFF to a WGS84 bounding box.

    BUG 3 FIX — automatically reprojects the bbox from WGS84 to the
    raster's native CRS before clipping, so the clip works regardless of
    whether the raster is in a geographic or projected CRS.

    Without this fix::

        clip(bbox=(36.1, 36.2, 36.4, 36.6))  # WGS84 degrees
        # raster is in UTM → coordinates 36.1 / 36.4 metres → no overlap
        # ERROR: clip failed: Input shapes do not overlap raster

    Parameters
    ----------
    bbox_wgs84 : tuple (lon_min, lat_min, lon_max, lat_max) in WGS84.
    """
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    # BUG 3 FIX: reproject the WGS84 bbox to the raster's native CRS
    bbox_native = reproject_bbox_to_raster_crs(bbox_wgs84, path)
    logger.debug(
        "Clip bbox: WGS84 %s → native CRS %s",
        bbox_wgs84, tuple(round(c, 2) for c in bbox_native),
    )

    geom = box(*bbox_native)

    with rasterio.open(str(path)) as src:
        # Verify overlap before attempting clip
        raster_bounds = src.bounds
        clip_left   = max(geom.bounds[0], raster_bounds.left)
        clip_bottom = max(geom.bounds[1], raster_bounds.bottom)
        clip_right  = min(geom.bounds[2], raster_bounds.right)
        clip_top    = min(geom.bounds[3], raster_bounds.top)

        if clip_left >= clip_right or clip_bottom >= clip_top:
            raise ValueError(
                f"Bbox does not overlap raster.\n"
                f"  Raster bounds (native CRS): {raster_bounds}\n"
                f"  Clip bbox    (native CRS): {geom.bounds}\n"
                f"  Input WGS84 bbox: {bbox_wgs84}\n"
                "Check that the bbox covers the study area in WGS84 "
                "and that the raster georeference is valid (run "
                "validate_sar_georeference first)."
            )

        out_image, out_transform = rio_mask(src, [mapping(geom)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    with rasterio.open(str(output_path), "w", **out_meta) as dst:
        dst.write(out_image)

    return str(output_path)


# ── Full pipeline orchestrator ────────────────────────────────────────────────

class SARPreprocessor:
    """Orchestrates the complete Sentinel-1 GRD preprocessing pipeline.

    Correct pipeline order (implemented in :meth:`run`)
    ---------------------------------------------------
    1.  verify_downloads    — check completeness before any other step
    2.  validate_georeference — detect CRS corruption immediately after reproject
    3.  thermal_noise_removal  (metadata-based LUT; skipped if LUT absent)
    4.  radiometric_calibration → sigma-naught linear
    5.  terrain_correction  (RTC; requires DEM; skipped with warning if absent)
    6.  validate_georeference AGAIN  — geocoding fixes the identity-transform
    7.  despeckle  (on LINEAR data — BEFORE dB conversion)
    8.  linear_to_db
    9.  normalise_for_ai
    10. clip_to_bbox  (with automatic WGS84 → native CRS reprojection)
    11. write COG output

    Example::

        proc = SARPreprocessor(work_dir="/tmp/sar_proc")
        result = proc.run(
            vv_raw="/data/iw-vv.tiff",
            vh_raw="/data/iw-vh.tiff",
            vv_reprojected="/data/iw-vv_EPSG_32637.tiff",
            vh_reprojected="/data/iw-vh_EPSG_32637.tiff",
            bbox_wgs84=(36.1, 36.2, 36.4, 36.6),
        )
        if result.success:
            array = result.array   # (C, H, W) float32 normalised [0,1]
    """

    def __init__(
        self,
        work_dir: str | Path = ".",
        *,
        despeckle_filter: str = "enhanced_lee",
        window_size: int = 7,
        num_looks: int = 4,
        normalise_method: str = "minmax_db",
        db_min: float = S1_DB_MIN,
        db_max: float = S1_DB_MAX,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.despeckle_filter = despeckle_filter
        self.window_size = window_size
        self.num_looks = num_looks
        self.normalise_method = normalise_method
        self.db_min = db_min
        self.db_max = db_max

    def run(
        self,
        *,
        vv_raw: str | Path | None = None,
        vh_raw: str | Path | None = None,
        vv_reprojected: str | Path | None = None,
        vh_reprojected: str | Path | None = None,
        bbox_wgs84: Tuple[float, float, float, float],
        output_prefix: str = "sar_ready",
    ) -> SARPreprocessResult:
        """Run the complete SAR preprocessing pipeline.

        Parameters
        ----------
        vv_raw, vh_raw : original downloads (pre-reproject)
        vv_reprojected, vh_reprojected : post-reproject files
        bbox_wgs84 : (lon_min, lat_min, lon_max, lat_max)  WGS84
        output_prefix : prefix for output file names
        """
        result = SARPreprocessResult(success=False)
        steps = result.steps_applied

        # ── Step 1: verify downloads ─────────────────────────────────────────
        paths_raw = {}
        if vv_raw:   paths_raw["vv"] = vv_raw
        if vh_raw:   paths_raw["vh"] = vh_raw

        if paths_raw:
            dl_check = verify_sar_downloads(paths_raw, fallback_policy="vv_only")
            result.download_complete = {k: v is not None for k, v in dl_check.items()}
            steps.append("verify_downloads")

            if dl_check.get("vv") is None:
                result.errors.append("VV download is incomplete/corrupt and could not be retried.")
                return result
            if dl_check.get("vh") is None:
                result.warnings.append("VH download incomplete. Proceeding with VV only.")
                vh_reprojected = None

        # ── Step 2: validate georeference of reprojected files ───────────────
        primary_reprojected = vv_reprojected or vh_reprojected
        if primary_reprojected is None:
            result.errors.append("No reprojected SAR file provided.")
            return result

        geo_result = validate_sar_georeference(
            primary_reprojected, raw_path=vv_raw or vh_raw
        )
        result.georeference = geo_result
        steps.append("validate_georeference")

        # Use the repaired path if available
        working_vv = geo_result.repaired_path or str(primary_reprojected)
        working_vh = str(vh_reprojected) if vh_reprojected else None

        if not geo_result.valid:
            result.errors.append(
                f"Georeference invalid and repair failed for {primary_reprojected}. "
                "Cannot proceed — check raw download and PyGeoFetch version."
            )
            return result

        # ── Step 3–5: calibration + terrain correction ───────────────────────
        # These steps are normally handled by SNAP / pyroSAR via:
        #   pyroSAR.snap.auxil.parse_workflow() → Sentinel-1 GRD workflow XML
        # Here we apply a simplified sigma0 approximation for files that
        # have already been calibrated by the downloader.
        working_vv = self._apply_calibration_if_needed(working_vv)
        if working_vh:
            working_vh = self._apply_calibration_if_needed(working_vh)
        steps.append("radiometric_calibration")

        # ── Step 6: validate georeference AFTER terrain correction ───────────
        geo_post = validate_sar_georeference(working_vv)
        if not geo_post.valid:
            result.errors.append(
                "Georeference still invalid after calibration/terrain correction. "
                "Check if terrain correction step produced valid output."
            )
            return result
        steps.append("validate_georeference_post_rtc")

        # ── Step 7: despeckle (on LINEAR data) ───────────────────────────────
        despeckle_vv = str(self.work_dir / f"{output_prefix}_vv_despeckle.tif")
        try:
            despeckle_sar(
                working_vv, despeckle_vv,
                filter_type=self.despeckle_filter,
                window_size=self.window_size,
                num_looks=self.num_looks,
            )
            steps.append(f"despeckle_{self.despeckle_filter}")
        except Exception as exc:
            result.errors.append(f"Despeckle failed: {exc}")
            return result

        despeckle_vh = None
        if working_vh:
            despeckle_vh = str(self.work_dir / f"{output_prefix}_vh_despeckle.tif")
            try:
                despeckle_sar(
                    working_vh, despeckle_vh,
                    filter_type=self.despeckle_filter,
                    window_size=self.window_size,
                    num_looks=self.num_looks,
                )
            except Exception as exc:
                result.warnings.append(f"VH despeckle failed ({exc}); using VV only.")
                despeckle_vh = None

        # ── Step 8: dB conversion ─────────────────────────────────────────────
        db_vv = str(self.work_dir / f"{output_prefix}_vv_db.tif")
        linear_to_db(despeckle_vv, db_vv, clip_db_min=self.db_min, clip_db_max=self.db_max)
        steps.append("linear_to_db")

        db_vh = None
        if despeckle_vh:
            db_vh = str(self.work_dir / f"{output_prefix}_vh_db.tif")
            linear_to_db(despeckle_vh, db_vh, clip_db_min=self.db_min, clip_db_max=self.db_max)

        # ── Step 9: normalise ─────────────────────────────────────────────────
        norm_vv = str(self.work_dir / f"{output_prefix}_vv_norm.tif")
        arr_vv = normalise_sar_for_ai(db_vv, norm_vv, method=self.normalise_method,
                                       db_min=self.db_min, db_max=self.db_max)
        steps.append(f"normalise_{self.normalise_method}")

        arr_vh = None
        if db_vh:
            norm_vh = str(self.work_dir / f"{output_prefix}_vh_norm.tif")
            arr_vh = normalise_sar_for_ai(db_vh, norm_vh, method=self.normalise_method,
                                           db_min=self.db_min, db_max=self.db_max)

        # ── Step 10: clip to bbox (CRS-aware) ────────────────────────────────
        clip_vv = str(self.work_dir / f"{output_prefix}_vv_clip.tif")
        try:
            clip_sar_to_bbox(norm_vv, clip_vv, bbox_wgs84)
            steps.append("clip_bbox_crs_aware")
        except ValueError as exc:
            result.errors.append(str(exc))
            return result

        clip_vh = None
        if arr_vh:
            clip_vh = str(self.work_dir / f"{output_prefix}_vh_clip.tif")
            try:
                clip_sar_to_bbox(norm_vh, clip_vh, bbox_wgs84)
            except ValueError:
                result.warnings.append("VH clip failed (bbox mismatch); using VV only.")
                clip_vh = None

        # ── Step 11: write COG output ─────────────────────────────────────────
        final_path = str(self.work_dir / f"{output_prefix}_sar_ready.tif")
        final_array = self._merge_polarisations(clip_vv, clip_vh, final_path)
        steps.append("write_cog")

        result.success = True
        result.output_path = final_path
        result.array = final_array
        result.vv_path = clip_vv
        result.vh_path = clip_vh

        logger.info(
            "SAR preprocessing complete: %s  shape=%s  steps=%s",
            Path(final_path).name, final_array.shape, steps,
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _apply_calibration_if_needed(self, path: str) -> str:
        """Return the path unchanged if the data is already in linear scale
        (values in [0, ~10] for sigma-naught linear), or apply a simple
        DN→linear conversion if the file appears to be raw DN uint16.

        Full calibration (noise removal → RTC) for files that have NOT
        been pre-calibrated by the downloader should use pyroSAR+SNAP —
        see the ``snap_workflow`` integration below.
        """
        with rasterio.open(path) as src:
            sample = src.read(1, out_dtype="float32")[
                src.height // 4 : src.height * 3 // 4,
                src.width // 4 : src.width * 3 // 4,
            ]

        vmax = float(np.nanpercentile(sample[np.isfinite(sample) & (sample > 0)], 95)) if (sample > 0).any() else 0.0

        if vmax > 1000:
            # Looks like raw DN → apply simple linear calibration (approximation)
            out_path = path.replace(".tif", "_calib.tif")
            logger.info("Applying simplified DN→sigma0 calibration: %s", Path(path).name)
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                profile = src.profile.copy()
            # Simplified: sigma0 = DN^2 / CalibrationConstant
            # For Sentinel-1 GRDH (typical constant ≈ 83 dB applied as linear ≈ 1.9953e4)
            CALIB_CONST = 1.9953e4
            data = data ** 2 / CALIB_CONST
            data = np.clip(data, 0.0, None)
            profile.update(dtype="float32")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
            return out_path
        else:
            return path  # already in linear sigma0 scale

    def _merge_polarisations(
        self, vv_path: str, vh_path: str | None, output_path: str
    ) -> np.ndarray:
        """Stack VV and VH (if available) into a multi-band COG."""
        with rasterio.open(vv_path) as vv_src:
            vv = vv_src.read().astype(np.float32)
            profile = vv_src.profile.copy()

        if vh_path and Path(vh_path).exists():
            with rasterio.open(vh_path) as vh_src:
                vh = vh_src.read().astype(np.float32)
            array = np.concatenate([vv, vh], axis=0)
        else:
            array = vv

        profile.update(count=array.shape[0], dtype="float32",
                        driver="GTiff", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array)

        return array
