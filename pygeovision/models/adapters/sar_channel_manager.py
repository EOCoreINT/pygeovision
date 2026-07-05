"""
pygeovision.models.adapters.sar_channel_manager
================================================
Channel management for multi-polarisation SAR data feeding into AI models
that expect a fixed number of input channels.

The problem
-----------
Most geospatial AI models (Prithvi, DINOv3, ChangeFormer) were pre-trained
on RGB or multispectral optical imagery with 3 or 6 channels in a well-
defined spectral order.  SAR data in dual-pol (VV + VH) gives 2 channels
with completely different physical meaning (radar backscatter, not surface
reflectance).

Feeding raw SAR channels as if they were optical bands causes silent
feature mismatches — the model 'sees' backscatter where it expects Blue,
Green, Red, NIR etc. and produces confused activations.

This module provides three strategies:

1.  ``sar_to_pseudo_rgb``  (for DINOv3 / any 3-channel model):
    Build a physically-meaningful 3-channel composite from VV, VH, and
    derived features.

2.  ``sar_to_hls_6ch``  (for Prithvi / HLS-trained models):
    Map SAR dual-pol features into the 6 expected HLS positions with
    physically motivated assignments.  This does NOT make the SAR data
    behave like optical data, but it ensures the feature at each channel
    position has some relationship to what an HLS band in that position
    conveys (e.g. VH in the NIR position because both distinguish
    vegetation from bare soil, though through different mechanisms).

3.  ``sar_learned_projection``  (stub for fine-tuned adapters):
    Placeholder for a learnable 1×1 convolution adapter trained on
    labelled SAR data (e.g. Sen1Floods11) using TerraTorch.

Physical SAR channel definitions
----------------------------------
For Sentinel-1 IW dual-pol (VV + VH) after dB conversion and [0,1]
normalisation:

    VV   — co-polarisation, sensitive to surface roughness, soil moisture,
            urban structure.  Strong in cities, dry farmland.
    VH   — cross-polarisation, sensitive to volume scattering (vegetation,
            forest).  Stronger in forested / vegetated areas.
    VV/VH ratio  — discriminates between urban (high VV, low VH → high ratio)
            and vegetation (lower ratio).  Analogous to NDVI in concept.
    VV - VH diff  — complementary to the ratio.
    geom_mean(VV,VH) — balanced representation of overall backscatter.
    (VV+VH)/2 — mean polarisation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("pygeovision.adapters.sar_channel_manager")


# ── Channel registry ──────────────────────────────────────────────────────────

SAR_CHANNELS = {
    "vv":        "Co-polarisation VV — surface roughness, soil moisture, urban",
    "vh":        "Cross-polarisation VH — volume scatter, vegetation",
    "vv_vh_ratio":    "VV/VH ratio — separates urban from vegetation",
    "vv_vh_diff":     "VV - VH difference",
    "vv_vh_geomean":  "sqrt(VV*VH) — geometric mean backscatter",
    "vv_vh_mean":     "Mean (VV+VH)/2",
}

# Prithvi/HLS expected 6-band positions and their optical meaning
HLS_PRITHVI_POSITIONS = {
    0: "Blue      (B02 / SR_B2)",
    1: "Green     (B03 / SR_B3)",
    2: "Red       (B04 / SR_B4)",
    3: "NIR       (B08 / SR_B5)",
    4: "SWIR-1    (B11 / SR_B6)",
    5: "SWIR-2    (B12 / SR_B7)",
}


# ── 3-channel pseudo-RGB (DINOv3 / ViT-style) ────────────────────────────────

def sar_to_pseudo_rgb(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    arrangement: str = "vv_vh_ratio",
) -> np.ndarray:
    """Build a 3-channel [0,1] composite from SAR dual-pol for RGB-based models.

    Parameters
    ----------
    vv, vh : np.ndarray  Shape (1, H, W) or (H, W), dtype float32, values [0,1].
    arrangement : str
        Which three channels to use:

        ``"vv_vh_ratio"`` (default) — ``[VV, VH, VV/VH]``
            Best for urban / damage detection (the ratio strongly
            discriminates between building facades and open ground).

        ``"vv_vh_diff"`` — ``[VV, VH, VV−VH]``
            Good for change detection pipelines.

        ``"vv_vh_geomean"`` — ``[VV, VH, sqrt(VV×VH)]``
            Balanced; good for land cover classification.

    Returns
    -------
    np.ndarray  Shape (3, H, W), float32, values [0, 1].
    """
    vv = _ensure_2d(vv)
    vh = _ensure_2d(vh)

    if arrangement == "vv_vh_ratio":
        # VV/VH ratio: clip to [0,1] by dividing by expected max ratio
        ratio = vv / (vh + 1e-8)
        ratio = np.clip(ratio / 4.0, 0.0, 1.0)  # normalise: ratio rarely exceeds 4x
        channels = [vv, vh, ratio]
    elif arrangement == "vv_vh_diff":
        diff = (vv - vh + 1.0) / 2.0  # shift to [0, 1]
        channels = [vv, vh, diff]
    elif arrangement == "vv_vh_geomean":
        geomean = np.sqrt(np.clip(vv * vh, 0.0, None))
        channels = [vv, vh, geomean]
    else:
        raise ValueError(
            f"Unknown arrangement '{arrangement}'. "
            "Use 'vv_vh_ratio', 'vv_vh_diff', or 'vv_vh_geomean'."
        )

    out = np.stack(channels, axis=0).astype(np.float32)
    assert out.shape[0] == 3
    return out


# ── 6-channel HLS mapping (Prithvi) ─────────────────────────────────────────

def sar_to_hls_6ch(
    vv: np.ndarray,
    vh: Optional[np.ndarray] = None,
    *,
    mapping: str = "physics_guided",
) -> np.ndarray:
    """Map SAR dual-pol features into Prithvi's 6-channel HLS input space.

    This function does NOT produce synthetic optical reflectance — the
    output is SAR-derived features arranged in a principled way that
    preserves physical meaning at each channel position.  It is NOT
    equivalent to fine-tuning (which modifies the model weights); it is
    a pre-processing step that works with the frozen pre-trained model.

    For best results on SAR, use this as a zero-shot baseline and then
    fine-tune with Sen1Floods11 + TerraTorch (see
    ``sar_prithvi_adapter.py``).

    Channel mapping (``physics_guided``)
    ------------------------------------
    Pos 0 (Blue  → volume scatter proxy) :  VH
    Pos 1 (Green → mean backscatter)     :  (VV+VH)/2
    Pos 2 (Red   → surface scatter)      :  VV
    Pos 3 (NIR   → vegetation proxy)     :  1−VH  (low VH→less vegetation→low NIR)
    Pos 4 (SWIR1 → moisture proxy)       :  1−VV  (low VV→moist soil→low SWIR)
    Pos 5 (SWIR2 → structural)           :  VV/VH ratio (urban discrimination)

    Parameters
    ----------
    vv : np.ndarray  Shape (1, H, W) or (H, W), float32, [0, 1].
    vh : np.ndarray | None  Same shape.  If None, VH is approximated
        from VV with a typical VV/VH ratio of 2.5 (urban/suburban default).
    mapping : str  ``"physics_guided"`` (default) or ``"replicate"``
        (simple: repeat VV and VH across 6 channels alternately).

    Returns
    -------
    np.ndarray  Shape (6, H, W), float32, [0, 1].
    """
    vv2d = _ensure_2d(vv)

    if vh is None:
        # VH approximation when cross-pol is missing
        # Typical VV/VH linear ratio ≈ 2.0–3.0 for mixed land cover
        logger.warning(
            "VH not provided — approximating VH = VV / 2.5. "
            "This is a rough fallback; accuracy will be degraded. "
            "Provide real VH for better results."
        )
        vh2d = vv2d / 2.5
    else:
        vh2d = _ensure_2d(vh)

    vv2d = np.clip(vv2d, 0.0, 1.0)
    vh2d = np.clip(vh2d, 0.0, 1.0)

    if mapping == "physics_guided":
        ratio = np.clip(vv2d / (vh2d + 1e-8) / 4.0, 0.0, 1.0)
        channels = [
            vh2d,                             # pos 0: Blue  → VH (volume/veg)
            (vv2d + vh2d) / 2.0,              # pos 1: Green → mean backscatter
            vv2d,                             # pos 2: Red   → VV (surface)
            np.clip(1.0 - vh2d, 0.0, 1.0),   # pos 3: NIR   → inverse-VH
            np.clip(1.0 - vv2d, 0.0, 1.0),   # pos 4: SWIR1 → inverse-VV
            ratio,                            # pos 5: SWIR2 → VV/VH ratio
        ]
    elif mapping == "replicate":
        channels = [vv2d, vh2d, vv2d, vh2d, vv2d, vh2d]
    else:
        raise ValueError(f"Unknown mapping '{mapping}'.")

    out = np.stack(channels, axis=0).astype(np.float32)
    assert out.shape == (6,) + vv2d.shape, f"Unexpected shape: {out.shape}"

    logger.info(
        "sar_to_hls_6ch: mapping=%s  shape=%s  vv=[%.3f,%.3f]  vh=[%.3f,%.3f]",
        mapping, out.shape,
        float(vv2d.min()), float(vv2d.max()),
        float(vh2d.min()), float(vh2d.max()),
    )
    return out


# ── Co-registration for change detection (ChangeFormer) ──────────────────────

def coregister_sar_pair(
    pre_path: str,
    post_path: str,
    output_dir: str,
    *,
    resampling_str: str = "bilinear",
) -> Tuple[str, str]:
    """Reproject the POST scene to exactly match the PRE scene grid.

    ChangeFormer and all other change-detection models require pixel-to-
    pixel spatial alignment — the PRE and POST arrays must have identical
    shape, transform, and CRS.  This function reprojects the POST raster
    onto the PRE grid using rasterio, guaranteeing alignment even when
    the two acquisitions come from different orbits or were processed with
    slightly different reproject parameters.

    Parameters
    ----------
    pre_path, post_path : str  Paths to the normalised, clipped SAR files.
    output_dir : str  Directory for the aligned output.

    Returns
    -------
    (pre_path, coregistered_post_path) : str, str
    """
    import rasterio
    from rasterio.enums import Resampling as _Resamp
    from rasterio.warp import reproject
    from pathlib import Path

    resamp = getattr(_Resamp, resampling_str, _Resamp.bilinear)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pre_path) as pre_src:
        pre_meta = pre_src.meta.copy()
        pre_transform = pre_src.transform
        pre_crs = pre_src.crs

    post_aligned_path = str(out_dir / (Path(post_path).stem + "_coregistered.tif"))
    with rasterio.open(post_path) as post_src:
        with rasterio.open(post_aligned_path, "w", **pre_meta) as dst:
            for i in range(1, post_src.count + 1):
                reproject(
                    source=rasterio.band(post_src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=post_src.transform,
                    src_crs=post_src.crs,
                    dst_transform=pre_transform,
                    dst_crs=pre_crs,
                    resampling=resamp,
                )

    logger.info("Co-registered POST to PRE grid: %s", post_aligned_path)
    return pre_path, post_aligned_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_sar_ai_input(
    array: np.ndarray,
    *,
    model: str = "prithvi",
    expected_channels: Optional[int] = None,
) -> dict:
    """Validate a prepared SAR array before passing it to an AI model.

    Checks dtype, range, NaN, and channel count.

    Returns
    -------
    dict with keys: valid (bool), errors (list), warnings (list), stats (dict)
    """
    result: dict = {"valid": True, "errors": [], "warnings": [], "stats": {}}

    if array.ndim != 3:
        result["errors"].append(f"Expected 3D array (C, H, W), got {array.ndim}D")
        result["valid"] = False
        return result

    c, h, w = array.shape
    n_channels = expected_channels or {"prithvi": 6, "dinov3": 3, "changeformer": None}.get(model)

    if n_channels and c != n_channels:
        result["errors"].append(f"Expected {n_channels} channels for {model}, got {c}")
        result["valid"] = False

    if array.dtype != np.float32:
        result["warnings"].append(f"Expected float32, got {array.dtype}; will auto-cast")

    nan_count = int(np.isnan(array).sum())
    if nan_count > 0:
        result["errors"].append(f"{nan_count} NaN values in array")
        result["valid"] = False

    vmin, vmax = float(np.nanmin(array)), float(np.nanmax(array))
    result["stats"] = {"min": vmin, "max": vmax, "mean": float(np.nanmean(array)),
                        "shape": array.shape, "dtype": str(array.dtype)}

    if vmax > 1.0 + 1e-4:
        result["warnings"].append(f"Values exceed 1.0 (max={vmax:.4f}); clipping recommended")
    if vmin < 0.0 - 1e-4:
        result["warnings"].append(f"Values below 0.0 (min={vmin:.4f}); clipping recommended")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Squeeze a (1, H, W) or (H, W) array to (H, W)."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected (H,W) or (1,H,W) array, got shape {arr.shape}")
