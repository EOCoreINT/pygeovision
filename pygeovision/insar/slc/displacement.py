"""
pygeovision.insar.slc.displacement
=====================================
Convert unwrapped SLC InSAR phase to ground displacement.

Physics
-------
Line-of-sight (LOS) displacement from unwrapped phase:

    d_LOS = (λ / 4π) × φ_unwrapped

where:
    λ = 0.05546576 m   (Sentinel-1 C-band wavelength)
    φ = unwrapped phase in radians (positive = motion toward satellite)

Decomposition into vertical and east-west components:
    Ascending  LOS vector:  [-sin(θ)sin(α),  sin(θ)cos(α), cos(θ)]
    Descending LOS vector:  [ sin(θ)sin(α), -sin(θ)cos(α), cos(θ)]

    where θ = incidence angle, α = satellite heading azimuth

With both ascending and descending pairs:
    d_V  = (d_asc·cos(θ_dsc) + d_dsc·cos(θ_asc)) / (...)
    d_EW = (d_asc - d_dsc) / (2·sin(θ)·sin(α))

Single pair gives only LOS — projection assumptions required for V/EW.

Convention:
    Positive LOS displacement → ground moved TOWARD the satellite
    In ascending pass: positive ~ uplift + westward
    In ascending pass: negative ~ subsidence + eastward
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("pygeovision.insar.slc.displacement")

# Physical constants
SENTINEL1_WAVELENGTH_M = 0.05546576     # metres (C-band, 5.405 GHz)
PHASE_TO_METRES        = SENTINEL1_WAVELENGTH_M / (4.0 * np.pi)

# Sentinel-1 IW mode typical parameters
# Incidence angle range: ~19°–46°, mid-swath per subswath:
SUBSWATH_INCIDENCE_DEG = {"IW1": 32.9, "IW2": 38.3, "IW3": 43.1}

# Sentinel-1 heading azimuth (degrees from North, clockwise)
# Ascending:  ~10° from North (near-north heading)
# Descending: ~170° from North
# Sentinel-1 satellite heading azimuth (degrees from North, clockwise)
# Ascending : satellite moves roughly northward  (~10°)
# Descending: satellite moves roughly southward  (~350°)
# NOTE: sin(10°) ≈ sin(170°) so 170° is NOT correct for the EW LOS component.
# The correct descending azimuth for the LOS east-west decomposition is ~350°
# (or equivalently -10°), giving sin(350°) ≈ -0.174 which is geometrically
# distinct from sin(10°) ≈ +0.174 and produces a well-conditioned inversion.
S1_HEADING_ASCENDING   = 10.0    # degrees from North  (northward track)
S1_HEADING_DESCENDING  = 350.0   # degrees from North  (southward track)


# ── Core conversion ───────────────────────────────────────────────────────────

def slc_phase_to_displacement(
    unwrapped_phase:      np.ndarray,
    coherence:            Optional[np.ndarray] = None,
    wavelength_m:         float = SENTINEL1_WAVELENGTH_M,
    incidence_angle_deg:  float = 38.3,
    coherence_threshold:  float = 0.3,
    output_path:          Optional[str] = None,
    geotransform:         Optional[tuple] = None,
    crs_wkt:              Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert unwrapped SLC InSAR phase to LOS displacement in metres.

    Parameters
    ----------
    unwrapped_phase : np.ndarray
        2D array of unwrapped phase in radians (from SNAPHU).
    coherence : np.ndarray, optional
        2D coherence array (0–1). Pixels below threshold are masked.
    wavelength_m : float
        SAR wavelength in metres. Default: Sentinel-1 C-band.
    incidence_angle_deg : float
        Mean incidence angle in degrees. Default: 38.3° (IW2 mid-swath).
    coherence_threshold : float
        Pixels with coherence below this value are set to NaN.
    output_path : str, optional
        If given, write the displacement GeoTIFF to this path.
        Requires geotransform and crs_wkt to be provided.
    geotransform : tuple, optional
        GDAL geotransform for output GeoTIFF.
    crs_wkt : str, optional
        CRS as WKT string for output GeoTIFF.

    Returns
    -------
    dict with keys:
        los_m         : LOS displacement (metres, + = toward satellite)
        vertical_m    : Estimated vertical displacement (metres, + = uplift)
        masked        : Boolean mask (True = invalid/low-coherence pixel)
        phase_rad     : Input unwrapped phase (copy)
        incidence_deg : Incidence angle used
        scale_factor  : λ/4π used for conversion
    """
    phase = np.asarray(unwrapped_phase, dtype=np.float64)

    # ── Coherence mask ────────────────────────────────────────────────────
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        # Resize coherence to match phase shape if needed (SNAP sometimes
        # produces slightly different sizes after multilooking)
        if coh.shape != phase.shape:
            from scipy.ndimage import zoom
            zy = phase.shape[0] / coh.shape[0]
            zx = phase.shape[1] / coh.shape[1]
            coh = zoom(coh, (zy, zx), order=1)
        mask = coh < coherence_threshold
    else:
        mask = np.zeros(phase.shape, dtype=bool)

    # ── LOS displacement ──────────────────────────────────────────────────
    scale = wavelength_m / (4.0 * np.pi)
    los_m = phase * scale
    los_m[mask] = np.nan

    # ── Vertical projection (single-pass approximation) ───────────────────
    # d_V ≈ d_LOS / cos(θ)
    # Valid assumption when horizontal displacement is small relative to
    # vertical, OR as a first-look estimate before dual-pass decomposition.
    theta_rad = np.deg2rad(incidence_angle_deg)
    vertical_m = los_m / np.cos(theta_rad)

    logger.info(
        "SLC displacement: scale=%.6f m/rad  incidence=%.1f°  "
        "LOS range=[%.3f, %.3f] m  masked=%.1f%%",
        scale, incidence_angle_deg,
        float(np.nanmin(los_m)),  float(np.nanmax(los_m)),
        100.0 * mask.sum() / mask.size,
    )

    result: Dict[str, np.ndarray] = {
        "los_m":          los_m.astype(np.float32),
        "vertical_m":     vertical_m.astype(np.float32),
        "masked":         mask,
        "phase_rad":      phase.astype(np.float32),
        "incidence_deg":  np.float32(incidence_angle_deg),
        "scale_factor":   np.float32(scale),
    }

    # ── Optional GeoTIFF export ───────────────────────────────────────────
    if output_path and geotransform and crs_wkt:
        _write_displacement_geotiff(
            result, output_path, geotransform, crs_wkt
        )
        logger.info("Displacement written: %s", output_path)

    return result


def los_to_vertical(
    los_m:               np.ndarray,
    incidence_angle_deg: float = 38.3,
) -> np.ndarray:
    """
    Project LOS displacement to approximate vertical displacement.

    Assumes horizontal displacement is negligible (valid for subsidence,
    slow volcanic deformation, and groundwater-driven compaction).
    For co-seismic deformation with large horizontal signal, use
    dual_pass_decomposition() instead.

    d_V = d_LOS / cos(θ)

    Parameters
    ----------
    los_m : np.ndarray
        LOS displacement in metres.
    incidence_angle_deg : float
        Radar incidence angle in degrees.

    Returns
    -------
    np.ndarray
        Vertical displacement in metres (positive = uplift).
    """
    return los_m / np.cos(np.deg2rad(incidence_angle_deg))


def dual_pass_decomposition(
    asc_los_m:   np.ndarray,
    dsc_los_m:   np.ndarray,
    asc_inc_deg: float = 38.3,
    dsc_inc_deg: float = 38.3,
    asc_head_deg: float = S1_HEADING_ASCENDING,
    dsc_head_deg: float = S1_HEADING_DESCENDING,
) -> Dict[str, np.ndarray]:
    """
    Decompose ascending + descending LOS into vertical and east-west.

    Requires co-registered ascending and descending displacement maps
    covering the same area and time period.

    The two-component model solves:
        [d_asc]   [cos(θ_a)  sin(θ_a)sin(α_a)] [d_V ]
        [d_dsc] = [cos(θ_d)  sin(θ_d)sin(α_d)] [d_EW]

    Parameters
    ----------
    asc_los_m : np.ndarray
        Ascending-pass LOS displacement (metres).
    dsc_los_m : np.ndarray
        Descending-pass LOS displacement (metres).
    asc_inc_deg : float
        Ascending incidence angle (degrees).
    dsc_inc_deg : float
        Descending incidence angle (degrees).
    asc_head_deg : float
        Ascending satellite heading (degrees from North, clockwise).
    dsc_head_deg : float
        Descending satellite heading (degrees from North, clockwise).

    Returns
    -------
    dict with keys:
        vertical_m  : Vertical displacement (metres, + = uplift)
        ew_m        : East-west displacement (metres, + = eastward)
        valid_mask  : True where both passes have valid data
    """
    θa = np.deg2rad(asc_inc_deg)
    θd = np.deg2rad(dsc_inc_deg)
    αa = np.deg2rad(asc_head_deg)
    αd = np.deg2rad(dsc_head_deg)

    # LOS unit vectors (vertical, east) components
    # Convention: d_LOS = d_V·cos(θ) + d_EW·sin(θ)·sin(α)
    a11 = np.cos(θa);  a12 = np.sin(θa) * np.sin(αa)
    a21 = np.cos(θd);  a22 = np.sin(θd) * np.sin(αd)

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-8:
        raise ValueError(
            "Ascending and descending LOS vectors are nearly parallel — "
            "decomposition is ill-conditioned. Check incidence/heading angles."
        )

    # 2×2 matrix inverse
    d_V  = ( a22 * asc_los_m - a12 * dsc_los_m) / det
    d_EW = (-a21 * asc_los_m + a11 * dsc_los_m) / det

    valid = np.isfinite(asc_los_m) & np.isfinite(dsc_los_m)

    d_V[~valid]  = np.nan
    d_EW[~valid] = np.nan

    logger.info(
        "Dual-pass decomposition: V=[%.3f, %.3f] m  EW=[%.3f, %.3f] m",
        float(np.nanmin(d_V)), float(np.nanmax(d_V)),
        float(np.nanmin(d_EW)), float(np.nanmax(d_EW)),
    )

    return {
        "vertical_m": d_V.astype(np.float32),
        "ew_m":       d_EW.astype(np.float32),
        "valid_mask": valid,
    }


def compute_deformation_rate(
    displacement_stack: list,           # List of np.ndarray (metres)
    time_days:          list,           # Days since first acquisition
    coherence_stack:    Optional[list] = None,
    coh_threshold:      float = 0.3,
) -> Dict[str, np.ndarray]:
    """
    Compute linear deformation rate from a time series of displacement maps.

    Fits d(t) = v·t + d₀ per pixel using weighted least squares.
    Coherence is used as weights when provided.

    Parameters
    ----------
    displacement_stack : list of np.ndarray
        N displacement maps in metres (must all have same shape).
    time_days : list of float
        Time in days since first acquisition for each map.
        Must have same length as displacement_stack.
    coherence_stack : list of np.ndarray, optional
        Per-epoch coherence maps for weighted inversion.
    coh_threshold : float
        Pixels with mean coherence below this are masked.

    Returns
    -------
    dict with keys:
        rate_m_per_year : Linear deformation rate (metres/year)
        intercept_m     : Displacement at t=0 (metres)
        residual_rms_mm : RMS residual per pixel (millimetres)
        coherent_mask   : True where fit is reliable
    """
    if len(displacement_stack) < 2:
        raise ValueError("Need at least 2 displacement maps for rate estimation.")

    stack = np.stack(displacement_stack, axis=0).astype(np.float64)  # (N, H, W)
    t     = np.array(time_days, dtype=np.float64)
    N, H, W = stack.shape

    # Build design matrix [t, 1]
    A = np.column_stack([t, np.ones(N)])   # (N, 2)

    # Per-pixel weighted least squares
    if coherence_stack is not None:
        W_coh = np.stack(coherence_stack, axis=0).astype(np.float64)
        W_coh = np.clip(W_coh, 0, 1)
    else:
        W_coh = np.ones_like(stack)

    # Vectorize: reshape to (N, H*W)
    d_flat   = stack.reshape(N, -1)
    w_flat   = W_coh.reshape(N, -1)

    rate_flat      = np.full(H * W, np.nan)
    intercept_flat = np.full(H * W, np.nan)
    rms_flat       = np.full(H * W, np.nan)

    for px in range(H * W):
        d_px = d_flat[:, px]
        w_px = w_flat[:, px]

        valid = np.isfinite(d_px)
        if valid.sum() < 2:
            continue

        dv = d_px[valid]
        wv = w_px[valid]
        tv = t[valid]

        # Weighted normal equations
        Aw = np.column_stack([tv, np.ones(len(tv))])
        W_diag = np.diag(wv)
        try:
            AtWA = Aw.T @ W_diag @ Aw
            AtWd = Aw.T @ W_diag @ dv
            coeff = np.linalg.solve(AtWA, AtWd)
        except np.linalg.LinAlgError:
            continue

        rate_flat[px]      = coeff[0]
        intercept_flat[px] = coeff[1]
        residuals          = dv - (coeff[0] * tv + coeff[1])
        rms_flat[px]       = np.sqrt(np.mean(residuals ** 2)) * 1000   # mm

    # Convert rate: m/day → m/year
    rate_flat *= 365.25

    # Coherence mask: mean coherence across time series
    mean_coh = W_coh.mean(axis=0).ravel()
    coherent = mean_coh >= coh_threshold

    rate_flat[~coherent] = np.nan

    return {
        "rate_m_per_year": rate_flat.reshape(H, W).astype(np.float32),
        "intercept_m":     intercept_flat.reshape(H, W).astype(np.float32),
        "residual_rms_mm": rms_flat.reshape(H, W).astype(np.float32),
        "coherent_mask":   coherent.reshape(H, W),
    }


# ── GeoTIFF output ────────────────────────────────────────────────────────────

def _write_displacement_geotiff(
    result:       Dict[str, np.ndarray],
    output_path:  str,
    geotransform: tuple,
    crs_wkt:      str,
) -> None:
    """Write LOS and vertical displacement as a multi-band GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS

        los_m  = result["los_m"]
        vert_m = result["vertical_m"]
        h, w   = los_m.shape

        gt = geotransform
        transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=2,
            dtype="float32",
            crs=CRS.from_wkt(crs_wkt),
            transform=transform,
            compress="deflate",
            nodata=np.nan,
        ) as dst:
            dst.write(los_m,  1)
            dst.write(vert_m, 2)
            dst.update_tags(
                1, description="LOS displacement (m, positive=toward satellite)"
            )
            dst.update_tags(
                2, description="Vertical displacement approx (m, positive=uplift)"
            )
    except ImportError:
        logger.warning("rasterio not installed — skipping GeoTIFF export")
