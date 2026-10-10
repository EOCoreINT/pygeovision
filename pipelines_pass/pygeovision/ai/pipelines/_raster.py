"""NaN-aware raster helpers shared by the pipelines.

Why this module exists
----------------------
Cloud- and shadow-masked pixels leave the preprocessing step as NaN
(``stack_and_prepare_bands`` fills them with NaN once reflectance scaling
is applied). Plain NumPy comparisons treat NaN as "False", so an index
pipeline that writes ``mask = ndsi > 0.4`` silently turns every masked
pixel into "not snow", and then divides by the *whole* raster, cloud
included. The same thing happens for "unburned", "not water" and
"no change". Every helper here carries an explicit *valid* mask so
that:

* an invalid pixel is written as nodata (255 for classes), never as a class;
* every percentage and area is computed over valid pixels only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

NODATA_CLASS = 255  # nodata value for every uint8 class / mask raster


def read_stack(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read every band as float32 ``(bands, rows, cols)`` plus the profile.

    A finite nodata value declared in the file is converted to NaN so that
    downstream code has exactly one invalid-pixel convention.
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == np.float32(nodata)] = np.nan
    return data, profile


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """True where *every* array is finite."""
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)`` clipped to [-1, 1]; NaN where undefined.

    No epsilon is added to the denominator: an epsilon silently turns an
    undefined pixel (both bands zero) into a confident 0.0.
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    s = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(s) & (np.abs(s) > 1e-6)
    out[ok] = (a[ok] - b[ok]) / s[ok]
    return np.clip(out, -1.0, 1.0)


def fraction(mask: np.ndarray, valid: np.ndarray) -> float | None:
    """Share of *valid* pixels where ``mask`` is True (None if no valid pixel)."""
    n = int(valid.sum())
    if n == 0:
        return None
    return float((mask & valid).sum()) / n


def area_ha(mask: np.ndarray, valid: np.ndarray, pixel_area_ha: float) -> float:
    """Area in hectares of ``mask`` restricted to valid pixels."""
    return float((mask & valid).sum()) * pixel_area_ha


def _clean_profile(profile: dict[str, Any], *, count: int, dtype: str, nodata: Any) -> dict[str, Any]:
    p = dict(profile)
    p.update(driver="GTiff", count=count, dtype=dtype, nodata=nodata, compress="lzw")
    if not p.get("tiled"):
        # blockxsize/blockysize are only legal on tiled files; a profile
        # copied from a striped source otherwise triggers GDAL warnings.
        p.pop("blockxsize", None)
        p.pop("blockysize", None)
    return p


def write_classes(
    path: str | Path,
    classes: np.ndarray,
    valid: np.ndarray,
    profile: dict[str, Any],
    descriptions: Sequence[str] | None = None,
) -> Path:
    """Write a single-band uint8 class raster, 255 where ``valid`` is False."""
    import rasterio

    arr = classes.astype(np.uint8, copy=True)
    arr[~valid] = NODATA_CLASS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_clean_profile(profile, count=1, dtype="uint8", nodata=NODATA_CLASS)) as dst:
        dst.write(arr[None, ...])
        if descriptions:
            dst.set_band_description(1, "; ".join(descriptions))
    return path


def write_float(
    path: str | Path,
    bands: np.ndarray | Sequence[np.ndarray],
    profile: dict[str, Any],
    descriptions: Sequence[str] | None = None,
) -> Path:
    """Write float32 band(s) with NaN as nodata."""
    import rasterio

    stack = np.asarray(bands, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_clean_profile(profile, count=stack.shape[0], dtype="float32", nodata=float("nan"))) as dst:
        dst.write(stack)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    return path
