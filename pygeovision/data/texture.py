"""
pygeovision.data.texture
=========================
Real Haralick Grey-Level Co-occurrence Matrix (GLCM) texture features.

Haralick, Shanmugam, Dinstein (1973), "Textural Features for Image
Classification" (IEEE Trans. Systems, Man, and Cybernetics) — the
original, standard reference for GLCM texture analysis, still the
default citation for texture features in remote sensing.

This module computes GENUINE GLCM features via `skimage.feature.
graycomatrix`/`graycoprops` — building the actual co-occurrence matrix
(the joint probability distribution of grey-level pairs at a given
offset/direction within a window) and deriving features from it. This is
NOT the same as a fast windowed-variance approximation to "contrast"
(which some libraries label "GLCM" for speed reasons without actually
building a co-occurrence matrix at all) — if your methods section says
"grey-level co-occurrence matrix", this module is what actually computes
that, not an approximation of its general spirit.

Two computation modes, both genuinely GLCM-based:
  - "block" (default): one GLCM per non-overlapping window — standard
    practice in remote sensing texture analysis (e.g. GRASS GIS
    r.texture, ENVI texture tools), and dramatically faster than a full
    sliding window since texture is inherently a neighbourhood property.
    Output resolution is coarser than the input (one value per block).
  - "sliding": one GLCM per pixel via a full sliding window — matches
    the input's native resolution, at real, substantial computational
    cost (O(H*W) GLCM computations). Use for small AOIs/test scenes, or
    when per-pixel texture resolution genuinely matters more than speed.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HARALICK_FEATURES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _require_skimage():
    try:
        from skimage.feature import graycomatrix, graycoprops
        return graycomatrix, graycoprops
    except ImportError:
        raise ImportError(
            "scikit-image required for real GLCM texture: pip install scikit-image"
        ) from None


def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("pip install rasterio") from None


def _quantise(band: np.ndarray, levels: int) -> np.ndarray:
    """Quantise a continuous band to `levels` discrete grey levels —
    GLCM operates on a discrete grey-level image, not continuous
    reflectance/backscatter values. NaN/invalid pixels map to level 0.
    """
    valid = np.isfinite(band)
    q = np.zeros(band.shape, dtype=np.uint8)
    if valid.any():
        d_min, d_max = np.nanmin(band[valid]), np.nanmax(band[valid])
        span = d_max - d_min
        if span > 1e-10:
            q[valid] = np.clip(
                ((band[valid] - d_min) / span * (levels - 1)), 0, levels - 1
            ).astype(np.uint8)
    return q


def _glcm_features_for_window(
    graycomatrix, graycoprops, window: np.ndarray, levels: int,
    distances: list[int], angles: list[float], features: tuple[str, ...],
) -> np.ndarray:
    """Real GLCM computation for a single window: build the actual
    co-occurrence matrix, average over the requested distances/angles,
    derive each requested Haralick feature from it."""
    glcm = graycomatrix(
        window, distances=distances, angles=angles, levels=levels,
        symmetric=True, normed=True,
    )
    out = np.empty(len(features), dtype=np.float32)
    for i, feat in enumerate(features):
        # graycoprops returns one value per (distance, angle) pair;
        # average across all of them for a single, direction-invariant
        # feature value per window (standard practice — a single
        # direction alone captures only anisotropic texture, not general
        # roughness).
        out[i] = graycoprops(glcm, feat).mean()
    return out


def compute_glcm_texture(
    band: np.ndarray,
    window: int = 7,
    mode: str = "block",
    levels: int = 32,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    features: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute real GLCM texture features from a single-band array.

    Args:
        band: 2D array (H, W) — a single band (e.g. NIR, or SAR VV/VH
            in dB — GLCM is defined per-band, not across multiple bands
            at once).
        window: Window size in pixels (odd number recommended).
        mode: "block" (default, fast — one GLCM per non-overlapping
            window, output is coarser than input) or "sliding" (one GLCM
            per pixel, matches input resolution, real computational cost).
        levels: Number of discrete grey levels to quantise to (standard
            GLCM practice: 16-64, since a full 256-level co-occurrence
            matrix is both slow and usually too sparse to be meaningful).
        distances: Pixel-pair distances to use (default [1]).
        angles: Pixel-pair angles in radians (default [0, pi/4, pi/2,
            3pi/4] — averaging across all 4 principal directions gives a
            rotation-invariant texture measure, standard practice unless
            you specifically want directional/anisotropic texture).
        features: Subset of HARALICK_FEATURES (default: all 6).

    Returns:
        Dict[feature_name, 2D array]. In "block" mode, each array has
        shape (H // window, W // window). In "sliding" mode, each array
        matches the input's (H, W) shape.
    """
    graycomatrix, graycoprops = _require_skimage()
    distances = distances or [1]
    angles = angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    features = tuple(features or HARALICK_FEATURES)

    for f in features:
        if f not in HARALICK_FEATURES:
            raise ValueError(f"Unknown GLCM feature {f!r}. Choose from: {HARALICK_FEATURES}")

    q = _quantise(band, levels)
    H, W = q.shape

    if mode == "block":
        out_h, out_w = H // window, W // window
        if out_h == 0 or out_w == 0:
            raise ValueError(
                f"window={window} is larger than the input ({H}x{W}) — "
                f"reduce window or use a bigger input."
            )
        results = {f: np.zeros((out_h, out_w), dtype=np.float32) for f in features}
        for r in range(out_h):
            for c in range(out_w):
                block = q[r * window:(r + 1) * window, c * window:(c + 1) * window]
                vals = _glcm_features_for_window(
                    graycomatrix, graycoprops, block, levels, distances, angles, features,
                )
                for i, f in enumerate(features):
                    results[f][r, c] = vals[i]
        return results

    elif mode == "sliding":
        half = window // 2
        padded = np.pad(q, half, mode="reflect")
        results = {f: np.zeros((H, W), dtype=np.float32) for f in features}
        for r in range(H):
            for c in range(W):
                win = padded[r:r + window, c:c + window]
                vals = _glcm_features_for_window(
                    graycomatrix, graycoprops, win, levels, distances, angles, features,
                )
                for i, f in enumerate(features):
                    results[f][r, c] = vals[i]
        return results

    else:
        raise ValueError(f"mode must be 'block' or 'sliding', got {mode!r}")


class GLCMTexture:
    """Real GLCM texture feature extraction from a raster band, with
    real georeferenced GeoTIFF I/O.

    Example::

        tex = GLCMTexture()
        result = tex.compute(
            "sentinel2_nir.tif", output_path="./texture.tif",
            window=7, mode="block", features=["contrast", "homogeneity", "energy"],
        )
    """

    def compute(
        self,
        input_path: str,
        output_path: str | None = None,
        band: int = 1,
        window: int = 7,
        mode: str = "block",
        levels: int = 32,
        distances: list[int] | None = None,
        angles: list[float] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute real GLCM texture and write a real, georeferenced
        multi-band GeoTIFF (one band per requested feature).

        In "block" mode, the output raster's pixel size is `window`
        times the input's, with a correspondingly adjusted transform —
        it stays correctly georeferenced, just at the block resolution.
        """
        rasterio = _require_rasterio()
        features = tuple(features or HARALICK_FEATURES)

        with rasterio.open(input_path) as src:
            arr = src.read(band).astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform

        results = compute_glcm_texture(
            arr, window=window, mode=mode, levels=levels,
            distances=distances, angles=angles, features=list(features),
        )

        stack = np.stack([results[f] for f in features])

        out_profile = profile.copy()
        out_profile.update(count=len(features), dtype="float32", compress="lzw")
        if mode == "block":
            # Output is coarser than input by `window` — scale the
            # transform accordingly so it stays correctly georeferenced,
            # not just visually similar.
            from rasterio.transform import Affine
            new_transform = transform * Affine.scale(window, window)
            out_profile.update(
                height=stack.shape[1], width=stack.shape[2], transform=new_transform,
            )

        out_path = output_path
        if out_path is None:
            p = pathlib.Path(input_path)
            out_path = str(p.parent / f"{p.stem}_glcm{p.suffix}")

        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(stack)
            for i, f in enumerate(features):
                dst.set_band_description(i + 1, f)
            dst.update_tags(method="GLCM (Haralick 1973)", window=str(window),
                             mode=mode, levels=str(levels))

        logger.info("GLCM texture (%s, window=%d, %d features) → %s",
                    mode, window, len(features), out_path)
        return {"success": True, "output_path": out_path, "features": list(features),
                "shape": stack.shape}
