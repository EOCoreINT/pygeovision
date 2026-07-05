"""
pygeovision.insar.coherence
============================
Coherence estimation and masking for InSAR quality control.

Coherence (γ, 0–1) measures the phase stability between two SAR acquisitions:
  γ = 1   → perfectly coherent (stable surface, reliable phase measurement)
  γ = 0   → fully decorrelated (vegetation, water, temporal change)

For GRD data (this module), amplitude coherence is used as a proxy.
For SLC data, the true interferometric coherence is computed from the complex signals.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("pygeovision.insar.coherence")


def estimate_coherence(
    pre_path:  str,
    post_path: str,
    output_path: str = "coherence.tif",
    window_size: int = 7,
) -> np.ndarray:
    """
    Estimate amplitude coherence from two co-registered SAR rasters.

    Parameters
    ----------
    pre_path, post_path : str
        Co-registered SAR rasters (float32, normalised [0, 1]).
    output_path : str
        Output GeoTIFF path.
    window_size : int
        Spatial averaging window (pixels × pixels).  Larger = smoother.

    Returns
    -------
    np.ndarray float32 coherence map in [0, 1].
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("pip install rasterio")

    from pygeovision.insar.interferogram import amplitude_coherence

    with rasterio.open(pre_path)  as src: pre  = src.read(1).astype("float32"); prof = src.profile.copy()
    with rasterio.open(post_path) as src: post = src.read(1).astype("float32")

    coh = amplitude_coherence(pre, post, window_size=window_size)

    prof.update(count=1, dtype="float32", nodata=-9999.0)
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(coh, 1)
        dst.update_tags(description="Amplitude coherence proxy (0=decorrelated, 1=coherent)")

    logger.info("Coherence: mean=%.3f  high(>0.7)=%.1f%%",
                float(np.nanmean(coh)), float((coh > 0.7).mean() * 100))
    return coh


def coherence_mask(
    coherence: np.ndarray,
    threshold: float = 0.4,
    output_path: Optional[str] = None,
    profile: Optional[dict] = None,
) -> np.ndarray:
    """
    Generate a binary mask where coherence exceeds the threshold.

    Low-coherence pixels (vegetated areas, water) are unreliable for
    phase unwrapping and displacement measurement.

    Returns
    -------
    np.ndarray uint8 binary mask: 1 = coherent (reliable), 0 = decorrelated.
    """
    mask = (coherence >= threshold).astype("uint8")

    if output_path and profile:
        try:
            import rasterio
            prof = profile.copy()
            prof.update(count=1, dtype="uint8", nodata=255)
            with rasterio.open(output_path, "w", **prof) as dst:
                dst.write(mask, 1)
        except Exception as exc:
            logger.warning("Could not save coherence mask: %s", exc)

    logger.info("Coherence mask: %.1f%% reliable pixels (threshold=%.2f)",
                float(mask.mean() * 100), threshold)
    return mask


def temporal_coherence(
    interferogram_stack: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Compute temporal coherence across a stack of interferograms.

    Parameters
    ----------
    interferogram_stack : np.ndarray shape (N, H, W)
        Stack of N wrapped phase rasters.

    Returns
    -------
    np.ndarray shape (H, W): mean coherence across all pairs.
    """
    from scipy.ndimage import uniform_filter

    n, h, w = interferogram_stack.shape
    coherences = []

    for i in range(n - 1):
        a = interferogram_stack[i].astype("float32")
        b = interferogram_stack[i + 1].astype("float32")
        # Amplitude proxy
        coh_i = np.abs(
            uniform_filter(a * b, size=window_size) /
            (np.sqrt(
                uniform_filter(a**2, size=window_size) *
                uniform_filter(b**2, size=window_size)
            ) + 1e-8)
        )
        coherences.append(coh_i)

    return np.nanmean(np.stack(coherences, axis=0), axis=0).astype("float32")
