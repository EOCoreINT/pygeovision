"""
pygeovision.utils.raster
========================
Raster I/O and band-math helpers.

Replaces two repeated patterns:

Pattern 1 — save ndarray as GeoTIFF::

    # Repeated in: obuasi, volta, smart_forest, coastal (4 notebooks)
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
    meta.update(count=1, dtype="float32", nodata=-9999.0)
    out_path = str(OUTPUT_DIR / "result.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr[np.newaxis])

Pattern 2 — NDVI / EVI / MNDWI band math::

    # Repeated in: obuasi, volta (ndvi); coastal (mndwi)
    nir = data[3]; red = data[2]; eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)

After::

    from pygeovision.utils import save_raster, band_ndvi, band_evi, band_mndwi

    save_raster(arr, ref_path, OUTPUT_DIR / "result.tif")
    ndvi = band_ndvi(stacked_tif)
    evi  = band_evi(stacked_tif)
    mndwi= band_mndwi(stacked_tif)
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6



def save_raster(
    array:       np.ndarray,
    reference:   str | pathlib.Path,
    output_path: str | pathlib.Path,
    nodata:      float = -9999.0,
    dtype:       str = "float32",
    compress:    str = "lzw",
) -> str:
    """
    Write a NumPy array to a GeoTIFF using spatial metadata from a reference file.

    Replaces the 7-line boilerplate::

        with rasterio.open(ref_path) as ref:
            meta = ref.meta.copy()
        meta.update(count=1, dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr[np.newaxis])

    Automatically handles:
      - 2-D arrays ``(H, W)`` → adds band dimension
      - 3-D arrays ``(C, H, W)`` → written as-is
      - dtype conversion
      - NaN → nodata replacement
      - LZW compression by default

    Args:
        array:       NumPy array ``(H, W)`` or ``(C, H, W)``.
        reference:   Path to a GeoTIFF whose CRS, transform, and extent
                     will be used. Array is cropped/padded to match.
        output_path: Destination ``.tif`` path.
        nodata:      NoData value. Default ``-9999.0``.
        dtype:       Output NumPy dtype. Default ``"float32"``.
        compress:    GDAL compression. Default ``"lzw"``.

    Returns:
        ``str(output_path)``.

    Example::

        from pygeovision.utils import save_raster

        save_raster(ndvi_arr, "s2_ready.tif", OUTPUT_DIR / "ndvi.tif")
        save_raster(risk_map, "s2_ready.tif", OUTPUT_DIR / "risk.tif",
                    dtype="uint8", nodata=255)
    """
    import rasterio

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure (C, H, W)
    arr = np.array(array, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim != 3:
        raise ValueError(f"array must be 2-D or 3-D, got shape {array.shape}")

    C, H, W = arr.shape

    # Replace NaN with nodata
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isfinite(arr), arr, nodata)

    with rasterio.open(str(reference)) as ref:
        meta = ref.meta.copy()
        ref_h, ref_w = ref.height, ref.width

    # Crop to reference dimensions
    arr = arr[:, :ref_h, :ref_w]

    meta.update(
        count    = arr.shape[0],
        dtype    = dtype,
        nodata   = nodata,
        compress = compress,
        height   = arr.shape[1],
        width    = arr.shape[2],
    )

    with rasterio.open(str(output_path), "w", **meta) as dst:
        dst.write(arr)

    logger.info("save_raster → %s  shape=%s", output_path.name, arr.shape)
    return str(output_path)


def _load_2d(source: str | np.ndarray, band_idx: int) -> tuple[np.ndarray, str | None]:
    """Load a single band from a stacked GeoTIFF or return array slice."""
    if isinstance(source, np.ndarray):
        arr = source.astype("float32")
        if arr.ndim == 2:
            return arr, None
        if arr.ndim == 3:
            return arr[band_idx - 1], None
        raise ValueError(f"Array must be 2-D or 3-D, got {arr.ndim}-D")

    import rasterio
    with rasterio.open(str(source)) as src:
        return src.read(band_idx).astype("float32"), str(source)


def _load_stack(source: str | np.ndarray) -> tuple[np.ndarray, str | None]:
    """Load all bands from a stacked GeoTIFF."""
    if isinstance(source, np.ndarray):
        return source.astype("float32"), None
    import rasterio
    with rasterio.open(str(source)) as src:
        return src.read().astype("float32"), str(source)


def _save_or_return(
    result:      np.ndarray,
    reference:   str | None,
    output_path: str | None,
    name:        str,
) -> np.ndarray | str:
    """Save result if output_path given, else return array."""
    if output_path is not None and reference is not None:
        return save_raster(result, reference, output_path)
    if output_path is not None:
        logger.warning(
            "%s: output_path set but source was an array — cannot infer spatial reference. "
            "Pass a file path as source to save GeoTIFF output.", name
        )
    return result


# ── Spectral index functions ──────────────────────────────────────────────────

def band_ndvi(
    source:     str | np.ndarray,
    red_band:   int = 3,
    nir_band:   int = 4,
    output_path: str | None = None,
) -> np.ndarray | str:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red) from a stacked raster.

    Replaces the inline band-math block repeated in obuasi and volta notebooks::

        nir = data[3]; red = data[2]; eps = 1e-6
        ndvi = (nir - red) / (nir + red + eps)
        ndvi = np.clip(ndvi, -1.0, 1.0)

    Default band order matches Sentinel-2 7-band stack
    ``[Blue, Green, Red, NIR, B8A, SWIR1, SWIR2]`` (1-indexed).

    Args:
        source:      Stacked GeoTIFF path or ``(C, H, W)`` float32 array.
        red_band:    1-based band index of Red channel. Default 3 (S2 B04).
        nir_band:    1-based band index of NIR channel. Default 4 (S2 B08).
        output_path: If set, saves result and returns the path.

    Returns:
        float32 array ``(H, W)`` in range ``[-1, 1]``, or path string if saved.

    Example::

        from pygeovision.utils import band_ndvi

        ndvi = band_ndvi("s2_ready.tif")
        ndvi = band_ndvi("s2_ready.tif", output_path="ndvi.tif")
    """
    data, ref = _load_stack(source)
    red  = data[red_band - 1]
    nir  = data[nir_band - 1]
    ndvi = (nir - red) / (nir + red + _EPS)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    logger.debug("band_ndvi: mean=%.3f std=%.3f", ndvi.mean(), ndvi.std())
    return _save_or_return(ndvi, ref, output_path, "band_ndvi")


def band_evi(
    source:      str | np.ndarray,
    blue_band:   int = 1,
    red_band:    int = 3,
    nir_band:    int = 4,
    G:  float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L:  float = 1.0,
    output_path: str | None = None,
) -> np.ndarray | str:
    """
    Compute EVI = G × (NIR − Red) / (NIR + C1×Red − C2×Blue + L).

    Enhanced Vegetation Index (Huete et al. 2002).
    Reduces atmospheric and canopy background effects compared to NDVI.
    Recommended for high-biomass areas like Ankasa Forest.

    Default band order matches Sentinel-2 7-band stack
    ``[Blue, Green, Red, NIR, B8A, SWIR1, SWIR2]`` (1-indexed).

    Args:
        source:      Stacked GeoTIFF path or ``(C, H, W)`` float32 array.
        blue_band:   1-based index of Blue. Default 1.
        red_band:    1-based index of Red. Default 3.
        nir_band:    1-based index of NIR. Default 4.
        G, C1, C2, L: EVI coefficients (MODIS standard defaults).
        output_path: If set, saves result and returns path.

    Returns:
        float32 array ``(H, W)`` clipped to ``[-1, 2]``, or path string.
    """
    data, ref = _load_stack(source)
    blue = data[blue_band - 1]
    red  = data[red_band  - 1]
    nir  = data[nir_band  - 1]
    evi  = G * (nir - red) / (nir + C1 * red - C2 * blue + L + _EPS)
    evi  = np.clip(evi, -1.0, 2.0)
    logger.debug("band_evi: mean=%.3f std=%.3f", evi.mean(), evi.std())
    return _save_or_return(evi, ref, output_path, "band_evi")


def band_mndwi(
    source:       str | np.ndarray,
    green_band:   int = 1,
    swir1_band:   int = 3,
    output_path:  str | None = None,
) -> np.ndarray | str:
    """
    Compute MNDWI = (Green − SWIR1) / (Green + SWIR1).

    Modified Normalised Difference Water Index (Xu 2006).
    Shoreline is the MNDWI = 0 contour.
    Better suppresses built-up and soil noise vs Gao NDWI.

    Default band order matches the 3-band S2 coastal download
    ``[Green(B03), NIR(B08), SWIR1(B11)]`` (1-indexed).

    Args:
        source:      Stacked GeoTIFF path or ``(C, H, W)`` float32 array.
        green_band:  1-based index of Green. Default 1.
        swir1_band:  1-based index of SWIR1. Default 3.
        output_path: If set, saves result and returns path.

    Returns:
        float32 array ``(H, W)`` in range ``[-1, 1]``, or path string.

    Example::

        from pygeovision.utils import band_mndwi

        mndwi = band_mndwi("s2_coastal.tif")
        water_mask = mndwi > 0.0
        # Shoreline = contour at mndwi == 0
    """
    data, ref = _load_stack(source)
    green = data[green_band - 1]
    swir1 = data[swir1_band - 1]
    mndwi = (green - swir1) / (green + swir1 + _EPS)
    mndwi = np.clip(mndwi, -1.0, 1.0)
    logger.debug("band_mndwi: mean=%.3f water=%.1f%%",
                 mndwi.mean(), (mndwi > 0).mean() * 100)
    return _save_or_return(mndwi, ref, output_path, "band_mndwi")


def band_nbr(
    source:      str | np.ndarray,
    nir_band:    int = 2,
    swir2_band:  int = 4,
    output_path: str | None = None,
) -> np.ndarray | str:
    """
    Compute NBR = (NIR − SWIR2) / (NIR + SWIR2).

    Normalised Burn Ratio — detects burned areas and forest disturbance.
    Default band order matches Landsat-9 4-band download
    ``[Red(B4), NIR(B5), SWIR1(B6), SWIR2(B7)]`` (1-indexed).
    """
    data, ref = _load_stack(source)
    nir   = data[nir_band   - 1]
    swir2 = data[swir2_band - 1]
    nbr   = (nir - swir2) / (nir + swir2 + _EPS)
    nbr   = np.clip(nbr, -1.0, 1.0)
    return _save_or_return(nbr, ref, output_path, "band_nbr")


def read_band(
    path:     str | pathlib.Path,
    band_idx: int = 1,
) -> np.ndarray:
    """
    Read a single band from a GeoTIFF as a float32 array.

    Args:
        path:     GeoTIFF file path.
        band_idx: 1-based band index.

    Returns:
        float32 array ``(H, W)``.
    """
    import rasterio
    with rasterio.open(str(path)) as src:
        return src.read(band_idx).astype("float32")


def read_meta(path: str | pathlib.Path) -> dict:
    """Return rasterio metadata dict for a GeoTIFF."""
    import rasterio
    with rasterio.open(str(path)) as src:
        return src.meta.copy()
