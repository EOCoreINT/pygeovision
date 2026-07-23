"""
pygeovision.insar.displacement
================================
Convert InSAR phase / amplitude change proxy to ground displacement.

True InSAR displacement (from SLC data):
    d_LOS = (λ / 4π) × φ_unwrapped

GRD amplitude proxy (this module default):
    displacement_proxy = amplitude_change_ratio
    Calibrated to centimetre scale using empirical scaling.

Line-of-sight (LOS) displacement can be decomposed into:
    • Vertical  : d_v ≈ d_LOS / cos(θ)
    • Horizontal: requires ascending + descending pair

Usage::

    from pygeovision.insar.displacement import phase_to_displacement, displacement_rate

    # From true InSAR unwrapped phase
    disp_m = phase_to_displacement(
        "unwrapped.tif", wavelength=0.055, incidence_angle=39.0,
        output_path="displacement_m.tif"
    )

    # From GRD amplitude proxy
    disp_proxy = phase_to_displacement(
        "amplitude_change.tif", mode="amplitude_proxy",
        output_path="displacement_proxy.tif"
    )

    # Time-series rate (mm/year)
    rate = displacement_rate(
        displacement_stack=["disp_2023_01.tif", ..., "disp_2024_01.tif"],
        dates=["2023-01-01", ..., "2024-01-01"],
        output_path="deformation_rate.tif"
    )
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("pygeovision.insar.displacement")

# Sentinel-1 C-band wavelength (metres)
SENTINEL1_WAVELENGTH = 0.05546576


def phase_to_displacement(
    phase_path: str,
    output_path: str = "displacement.tif",
    mode: str = "slc_phase",          # "slc_phase" | "amplitude_proxy"
    wavelength: float = SENTINEL1_WAVELENGTH,
    incidence_angle_deg: float = 39.0,
    vertical_only: bool = False,
) -> np.ndarray:
    """
    Convert phase / amplitude change to displacement map.

    Parameters
    ----------
    phase_path : str
        Unwrapped phase raster (SLC mode) OR amplitude change raster (proxy mode).
    mode : str
        "slc_phase" — true InSAR: d_LOS = λ/(4π) × φ_unwrapped
        "amplitude_proxy" — GRD mode: empirical scaling of amplitude change ratio
    wavelength : float
        SAR wavelength in metres.  Sentinel-1 C-band = 0.05546576 m.
    incidence_angle_deg : float
        Mean incidence angle of the acquisition (degrees).
    vertical_only : bool
        If True, project LOS to vertical component.

    Returns
    -------
    np.ndarray float32 displacement in metres (LOS or vertical).
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("pip install rasterio")

    with rasterio.open(phase_path) as src:
        phase = src.read(1).astype("float32")
        profile = src.profile.copy()

    if mode == "slc_phase":
        # True InSAR formula: d_LOS [m] = λ/(4π) × φ_unwrapped [rad]
        displacement = (wavelength / (4 * np.pi)) * phase
        logger.info("SLC phase → LOS displacement  λ=%.4f m", wavelength)
    else:
        # Amplitude proxy: scale change ratio to empirical displacement range
        # The amplitude change ratio (–1 to +1) is mapped to ±50 cm range
        # This is a rough proxy; true displacement requires SLC InSAR.
        displacement = phase * 0.5   # ±0.5 m range for ratio in [–1, 1]
        logger.info("Amplitude proxy displacement (empirical scaling, not true InSAR)")

    # Project LOS → vertical
    if vertical_only:
        cos_theta    = np.cos(np.radians(incidence_angle_deg))
        displacement = displacement / (cos_theta + 1e-8)
        logger.info("LOS → vertical projection  θ=%.1f°", incidence_angle_deg)

    # Save
    profile.update(count=1, dtype="float32", nodata=-9999.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(displacement.astype('float32'), 1)
        dst.update_tags(
            mode=mode,
            wavelength_m=str(wavelength),
            incidence_angle_deg=str(incidence_angle_deg),
            units="metres",
        )

    stats = {
        "min_m":  round(float(np.nanmin(displacement)), 4),
        "max_m":  round(float(np.nanmax(displacement)), 4),
        "mean_m": round(float(np.nanmean(displacement)), 4),
        "std_m":  round(float(np.nanstd(displacement)), 4),
    }
    logger.info("Displacement: min=%.3f m  max=%.3f m  mean=%.3f m",
                stats["min_m"], stats["max_m"], stats["mean_m"])
    return displacement


def displacement_rate(
    displacement_paths: list[str],
    dates:              list[str],
    output_path:        str = "deformation_rate_mm_yr.tif",
    mask_path:          str | None = None,
) -> np.ndarray:
    """
    Estimate pixel-wise linear deformation rate (mm/year) from a displacement stack.

    Parameters
    ----------
    displacement_paths : list[str]
        Ordered list of displacement rasters (metres).
    dates : list[str]
        ISO date strings for each raster.
    output_path : str
        Output GeoTIFF path (values in mm/year).
    mask_path : str | None
        Optional coherence mask.

    Returns
    -------
    np.ndarray float32 rate in mm/year.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("pip install rasterio")

    assert len(displacement_paths) == len(dates), "paths and dates must have same length"
    assert len(dates) >= 2, "need at least 2 displacement maps for rate estimation"

    # Convert dates to decimal years
    def _decimal_year(date_str: str) -> float:
        dt = datetime.fromisoformat(date_str)
        start = datetime(dt.year, 1, 1)
        end   = datetime(dt.year + 1, 1, 1)
        return dt.year + (dt - start).days / (end - start).days

    years = np.array([_decimal_year(d) for d in dates])

    # Load displacement stack
    with rasterio.open(displacement_paths[0]) as src:
        profile = src.profile.copy()
        H, W    = src.height, src.width

    stack = np.zeros((len(displacement_paths), H, W), dtype="float32")
    for i, path in enumerate(displacement_paths):
        with rasterio.open(path) as src:
            stack[i] = src.read(1).astype("float32")

    # Optional coherence mask
    mask = None
    if mask_path:
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(bool)

    # Pixel-wise linear regression (vectorised)
    x     = years - years.mean()
    xsum  = (x**2).sum()
    rate  = np.zeros((H, W), dtype="float32")

    # Batch column-wise regression
    for row in range(H):
        y_row  = stack[:, row, :]           # (T, W)
        slopes = (x @ y_row) / (xsum + 1e-10)  # (W,)
        rate[row, :] = slopes

    # Convert m/year → mm/year
    rate_mm = rate * 1000.0

    if mask is not None:
        rate_mm[~mask] = np.nan

    profile.update(count=1, dtype="float32", nodata=-9999.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.nan_to_num(rate_mm, nan=-9999.0).astype('float32'), 1)
        dst.update_tags(units="mm/year", n_acquisitions=len(dates))

    logger.info("Deformation rate: min=%.1f  max=%.1f  mean=%.1f mm/yr",
                float(np.nanmin(rate_mm)), float(np.nanmax(rate_mm)),
                float(np.nanmean(rate_mm)))
    return rate_mm
