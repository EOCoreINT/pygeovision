"""
pygeovision.data.processors.sar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentinel-1 GRD SAR preprocessing pipeline.

Correct processing order (CRITICAL):
  S0: verify_sar_downloads()       — check all tiles readable
  S1: validate_sar_georeference()  — detect identity transform, recover via GCPs
  S6: despeckle_sar()              — Lee/boxcar filter on LINEAR power data
  S7: linear_to_db()               — convert to dB AFTER despeckle
  S8: normalise_sar_for_ai()       — normalise [0,1] with fixed physical bounds
  S9: clip_sar_to_bbox()           — CRS-aware clip to study area

Bug fixes (Jul 2026):
  BUG 1: validate_sar_georeference() now attempts GCP recovery when identity
          transform is detected. geo.repaired_path is populated on success.
          Previously returned None for every pygeofetch-downloaded file.
  BUG 2: check_download_complete() moved to georeference.py, imported here.
  BUG 3: clip_sar_to_bbox() reprojects WGS84 bbox to raster CRS before clipping.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import calculate_default_transform
from rasterio.warp import reproject

# Import the fixed validator (this is the key fix — replaces old broken validator)
from pygeovision.data.validators.georeference import (
    GeoreferenceResult,
    validate_georeference,
    validate_sar_georeference,
    check_download_complete,
    reproject_bbox_to_raster_crs,
)

logger = logging.getLogger(__name__)

# ── Physical constants for Sentinel-1 GRD ────────────────────────────────────
S1_DB_MIN    = -35.0   # minimum dB value (noise floor for VH)
S1_DB_MAX    =   5.0   # maximum dB value (urban double-bounce)
S1_LINEAR_MIN = 10 ** (S1_DB_MIN / 10)
S1_LINEAR_MAX = 10 ** (S1_DB_MAX / 10)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SARPreprocessResult:
    """Result of a single SAR preprocessing step or verify_sar_downloads()."""
    success:  bool
    path:     Optional[str] = None
    errors:   List[str]     = field(default_factory=list)
    warnings: List[str]     = field(default_factory=list)
    shape:    Optional[Tuple[int, ...]] = None
    stats:    Dict[str, float]          = field(default_factory=dict)


# ── Speckle filters ───────────────────────────────────────────────────────────

def _boxcar(data: np.ndarray, window_size: int) -> np.ndarray:
    """Simple boxcar (mean) filter — fastest, least edge-preserving."""
    from scipy.ndimage import uniform_filter
    out = np.empty_like(data)
    for b in range(data.shape[0]):
        out[b] = uniform_filter(data[b].astype('float64'), size=window_size).astype('float32')
    return out


def _enhanced_lee(data: np.ndarray, window_size: int, num_looks: int = 4) -> np.ndarray:
    """
    Enhanced Lee filter — standard choice for Sentinel-1 SAR.
    Operates on LINEAR power data. Do NOT pass dB values.
    """
    from scipy.ndimage import uniform_filter
    out = np.empty_like(data)
    w = window_size
    for b in range(data.shape[0]):
        band = data[b].astype('float64')
        mean = uniform_filter(band, w)
        mean_sq = uniform_filter(band ** 2, w)
        var_local = np.maximum(mean_sq - mean ** 2, 0.0)
        var_noise = mean ** 2 / max(num_looks, 1)
        weight = np.where(
            var_local + var_noise > 0,
            var_local / (var_local + var_noise),
            0.0
        )
        out[b] = (mean + weight * (band - mean)).astype('float32')
    return out


def _refined_lee(data: np.ndarray, window_size: int, num_looks: int = 4) -> np.ndarray:
    """
    Refined Lee filter — better edge preservation, slower than enhanced Lee.
    """
    # For now falls back to enhanced Lee — a full refined-Lee implementation
    # requires directional window selection which is out of scope for this fix.
    return _enhanced_lee(data, window_size, num_looks)


# ── Core processing functions ─────────────────────────────────────────────────

def verify_sar_downloads(paths: List[str]) -> List[SARPreprocessResult]:
    """
    S0: Verify that all downloaded SAR GeoTIFFs are complete and readable.

    Args:
        paths: List of file paths to check.

    Returns:
        List of SARPreprocessResult, one per input path.
    """
    results = []
    for path in paths:
        from pygeovision.data.validators.georeference import check_download_complete
        dlc = check_download_complete(str(path))
        r = SARPreprocessResult(
            success  = dlc["complete"],
            path     = str(path) if dlc["complete"] else None,
            errors   = dlc["errors"],
            warnings = [],
            stats    = {"file_size_mb": dlc["file_size_mb"],
                        "readable_tiles": dlc["readable_tiles"],
                        "total_tiles": dlc["total_tiles"]},
        )
        if not dlc["complete"]:
            r.warnings.append(
                f"Incomplete: {dlc['readable_tiles']}/{dlc['total_tiles']} tiles readable"
            )
        results.append(r)
    return results


def despeckle_sar(
    input_path: str,
    output_path: str,
    filter_type: str = "enhanced_lee",
    window_size: int = 7,
    num_looks:   int = 4,
) -> str:
    """
    S6: Apply speckle filter to SAR data in LINEAR power space.

    CRITICAL: This must run BEFORE dB conversion. Running on dB data
    produces incorrect noise suppression because dB values are logarithmic.

    Args:
        input_path:  Path to input GeoTIFF (linear power values).
        output_path: Path for output GeoTIFF.
        filter_type: 'enhanced_lee' | 'refined_lee' | 'boxcar'.
        window_size: Filter window size (pixels). 7 is standard for S-1.
        num_looks:   Equivalent number of looks. 4 is standard for S-1 GRD.

    Returns:
        output_path on success.
    """
    filter_fn = {
        "enhanced_lee": _enhanced_lee,
        "refined_lee":  _refined_lee,
        "boxcar":       lambda d, w, **kw: _boxcar(d, w),
    }.get(filter_type, _enhanced_lee)

    with rasterio.open(input_path) as src:
        data = src.read().astype("float32")
        meta = src.meta.copy()

    # Ensure values are positive (linear power must be > 0)
    data = np.maximum(data, 1e-10)

    filtered = filter_fn(data, window_size, num_looks=num_looks)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(filtered)

    logger.debug(f"despeckle_sar: {filter_type}, window={window_size} → {Path(output_path).name}")
    return output_path


def linear_to_db(
    input_path:   str,
    output_path:  str,
    clip_db_min:  float = S1_DB_MIN,
    clip_db_max:  float = S1_DB_MAX,
    epsilon:      float = 1e-10,
) -> str:
    """
    S7: Convert linear power SAR data to dB scale.

    Formula: dB = 10 * log10(max(linear, epsilon))
    Applied AFTER despeckle_sar().

    Args:
        input_path:  Path to linear-power GeoTIFF (output of despeckle_sar).
        output_path: Path for dB GeoTIFF.
        clip_db_min: Minimum dB value to clip to (default -35 dB).
        clip_db_max: Maximum dB value to clip to (default +5 dB).
        epsilon:     Floor to prevent log(0).

    Returns:
        output_path on success.
    """
    with rasterio.open(input_path) as src:
        data = src.read().astype("float32")
        meta = src.meta.copy()

    db = 10.0 * np.log10(np.maximum(data, epsilon))
    db = np.clip(db, clip_db_min, clip_db_max)

    meta.update(dtype="float32")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(db)

    logger.debug(
        f"linear_to_db: range [{db.min():.1f}, {db.max():.1f}] dB → {Path(output_path).name}"
    )
    return output_path


def normalise_sar_for_ai(
    input_path:  str,
    output_path: str,
    method:      str = "minmax_db",
    db_min:      float = S1_DB_MIN,
    db_max:      float = S1_DB_MAX,
) -> str:
    """
    S8: Normalise SAR data to [0, 1] for AI/ML input.

    Args:
        input_path:  Path to dB GeoTIFF (output of linear_to_db).
        output_path: Path for normalised GeoTIFF.
        method:      'minmax_db'  — use fixed physical dB bounds (recommended).
                     'percentile' — use per-scene 2nd/98th percentile (avoids
                                    stretching to outliers, but less cross-scene
                                    comparable).
        db_min:      Lower dB bound for minmax_db normalisation.
        db_max:      Upper dB bound for minmax_db normalisation.

    Returns:
        output_path on success.
    """
    with rasterio.open(input_path) as src:
        data = src.read().astype("float32")
        meta = src.meta.copy()

    if method == "minmax_db":
        normed = (data - db_min) / (db_max - db_min)
        normed = np.clip(normed, 0.0, 1.0)
    elif method == "percentile":
        out = np.empty_like(data)
        for b in range(data.shape[0]):
            p2  = np.nanpercentile(data[b], 2)
            p98 = np.nanpercentile(data[b], 98)
            denom = max(p98 - p2, 1e-6)
            out[b] = np.clip((data[b] - p2) / denom, 0.0, 1.0)
        normed = out
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")

    meta.update(dtype="float32", nodata=None)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(normed)

    logger.debug(
        f"normalise_sar_for_ai ({method}): [{normed.min():.4f}, {normed.max():.4f}] "
        f"→ {Path(output_path).name}"
    )
    return output_path


def clip_sar_to_bbox(
    input_path:  str,
    output_path: str,
    bbox_wgs84:  Tuple[float, float, float, float],
) -> str:
    """
    S9: Clip SAR raster to a WGS84 bounding box, auto-reprojecting the bbox
    to the raster's CRS before clipping (BUG 3 fix).

    Args:
        input_path:  Path to normalised GeoTIFF.
        output_path: Path for clipped GeoTIFF.
        bbox_wgs84:  (lon_min, lat_min, lon_max, lat_max) in WGS84.

    Returns:
        output_path on success.

    Raises:
        ValueError: If the bbox does not overlap the raster extent.
    """
    from rasterio.windows import from_bounds as window_from_bounds

    # Reproject bbox to raster CRS (BUG 3 fix)
    raster_bbox = reproject_bbox_to_raster_crs(bbox_wgs84, input_path)

    with rasterio.open(input_path) as src:
        # Check overlap
        r = src.bounds
        clip_box = raster_bbox
        if (clip_box[2] <= r.left or clip_box[0] >= r.right or
                clip_box[3] <= r.bottom or clip_box[1] >= r.top):
            raise ValueError(
                f"Clip bbox {clip_box} does not overlap raster bounds "
                f"({r.left:.1f}, {r.bottom:.1f}, {r.right:.1f}, {r.top:.1f})"
            )

        # Intersect with raster bounds
        x_min = max(clip_box[0], r.left)
        y_min = max(clip_box[1], r.bottom)
        x_max = min(clip_box[2], r.right)
        y_max = min(clip_box[3], r.top)

        win  = window_from_bounds(x_min, y_min, x_max, y_max, src.transform)
        data = src.read(window=win)
        new_transform = src.window_transform(win)
        meta = src.meta.copy()
        meta.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=new_transform,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)

    logger.debug(
        f"clip_sar_to_bbox: {data.shape} → {Path(output_path).name}"
    )
    return output_path


# ── assert_valid_georeference ────────────────────────────────────────────────

def assert_valid_georeference(path: str) -> None:
    """Raise ValueError if georeference is invalid and unrecoverable."""
    result = validate_sar_georeference(path)
    if not result.valid and result.repaired_path is None:
        raise ValueError(
            f"Invalid georeference in {Path(path).name}: {result.errors}"
        )


# ── High-level SARPreprocessor class ─────────────────────────────────────────

class SARPreprocessor:
    """
    High-level SAR preprocessing pipeline.

    Runs the full S0→S9 pipeline with automatic GCP recovery (BUG 1 fix),
    download verification (BUG 2 fix), and CRS-aware clipping (BUG 3 fix).

    Usage:
        preprocessor = SARPreprocessor(work_dir='./work', bbox_wgs84=ODAW_BBOX)
        result = preprocessor.run(raw_path, output_path, label='2024_jun')
    """

    def __init__(
        self,
        work_dir:      str = "./sar_work",
        bbox_wgs84:    Optional[Tuple[float, float, float, float]] = None,
        filter_type:   str = "enhanced_lee",
        window_size:   int = 7,
        num_looks:     int = 4,
        db_min:        float = S1_DB_MIN,
        db_max:        float = S1_DB_MAX,
    ):
        self.work_dir    = Path(work_dir)
        self.bbox_wgs84  = bbox_wgs84
        self.filter_type = filter_type
        self.window_size = window_size
        self.num_looks   = num_looks
        self.db_min      = db_min
        self.db_max      = db_max
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        raw_path:    str,
        output_path: str,
        label:       str = "scene",
    ) -> Optional[str]:
        """
        Run full S0→S9 preprocessing pipeline.

        Returns output_path on success, None on failure.
        """
        raw_path    = Path(raw_path)
        output_path = Path(output_path)

        if not raw_path.exists():
            logger.error(f"[{label}] Raw file not found: {raw_path}")
            return None

        stem = raw_path.stem

        # S0: verify download
        dlc = check_download_complete(str(raw_path))
        if not dlc["complete"]:
            if dlc["readable_tiles"] == 0:
                logger.error(f"[{label}] Zero readable tiles — skipping")
                return None
            logger.warning(
                f"[{label}] Incomplete download "
                f"({dlc['readable_tiles']}/{dlc['total_tiles']} tiles)"
            )

        # S1: georeference validation + GCP recovery
        geo = validate_sar_georeference(str(raw_path))
        working = geo.repaired_path or str(raw_path)

        if not geo.valid and geo.repaired_path is None:
            logger.error(f"[{label}] Georeference invalid and unrecoverable: {geo.errors}")
            return None

        if geo.recovered:
            logger.info(f"[{label}] Using GCP-recovered file: {geo.repaired_path}")

        # S6: despeckle on linear data
        desp_path = str(self.work_dir / f"{stem}_despeckle.tif")
        try:
            despeckle_sar(
                working, desp_path,
                filter_type=self.filter_type,
                window_size=self.window_size,
                num_looks=self.num_looks,
            )
        except Exception as e:
            logger.warning(f"[{label}] Despeckle failed ({e}), using raw file")
            shutil.copy2(working, desp_path)

        # S7: convert to dB
        db_path = str(self.work_dir / f"{stem}_db.tif")
        linear_to_db(desp_path, db_path, self.db_min, self.db_max)

        # S8: normalise
        norm_path = str(self.work_dir / f"{stem}_norm.tif")
        normalise_sar_for_ai(db_path, norm_path, method="minmax_db",
                             db_min=self.db_min, db_max=self.db_max)

        # S9: clip to bbox
        if self.bbox_wgs84:
            try:
                clip_sar_to_bbox(norm_path, str(output_path), self.bbox_wgs84)
            except ValueError as e:
                logger.warning(f"[{label}] Clip failed ({e}), using full extent")
                shutil.copy2(norm_path, str(output_path))
        else:
            shutil.copy2(norm_path, str(output_path))

        # Final read to confirm output is valid
        with rasterio.open(str(output_path)) as src:
            arr = src.read()

        logger.info(
            f"[{label}] Preprocessing complete: {arr.shape} "
            f"[{arr.min():.4f}, {arr.max():.4f}] → {output_path.name}"
        )
        return str(output_path)