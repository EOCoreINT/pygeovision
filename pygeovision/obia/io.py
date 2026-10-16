"""Reading rasters into the (bands, rows, cols) layout used across ``obia``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RasterData:
    """A multi-band image plus the georeferencing needed to export results."""

    image: np.ndarray                 # (bands, rows, cols), float32
    band_names: list[str]
    transform: Any = None             # rasterio Affine or None
    crs: Any = None
    nodata: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[2]

    @property
    def pixel_area(self) -> float | None:
        """Pixel area in CRS units² (m² for projected CRS), if known."""
        if self.transform is None:
            return None
        return abs(self.transform.a * self.transform.e)


def load_raster(path: str, bands: list[int] | None = None,
                band_names: list[str] | None = None) -> RasterData:
    """Read a GeoTIFF into a :class:`RasterData`.

    ``bands`` are 1-based indices (default: all). Band names come from
    ``band_names`` if given, else from the file's band descriptions, else
    ``b1, b2, ...``.
    """
    import rasterio

    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        arr = src.read(idx).astype(np.float32)
        names = band_names
        if names is None:
            desc = [src.descriptions[i - 1] for i in idx]
            names = [d if d else f"b{i}" for d, i in zip(desc, idx)]
        if len(names) != len(idx):
            raise ValueError(f"{len(names)} band names for {len(idx)} bands")
        return RasterData(arr, list(names), src.transform, src.crs, src.nodata)


def as_raster_data(image, band_names=None, transform=None, crs=None) -> RasterData:
    """Accept a path, a :class:`RasterData`, or a (bands, rows, cols) array."""
    if isinstance(image, RasterData):
        return image
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        return load_raster(str(image), band_names=band_names)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError("image must be (bands, rows, cols) or (rows, cols)")
    names = list(band_names) if band_names else [f"b{i + 1}" for i in range(arr.shape[0])]
    if len(names) != arr.shape[0]:
        raise ValueError(f"{len(names)} band names for {arr.shape[0]} bands")
    return RasterData(arr, names, transform, crs)


def valid_mask(raster) -> np.ndarray:
    """Boolean ``(rows, cols)`` mask of pixels that hold data.

    A pixel is invalid when any band is NaN or infinite (what cloud and shadow masking leaves behind), or when every band
    equals the raster's declared ``nodata`` value. ``raster`` is a :class:`RasterData` or a ``(bands, rows, cols)`` array.
    """
    if isinstance(raster, RasterData):
        arr, nodata = raster.image, raster.nodata
    else:
        arr, nodata = np.asarray(raster), None
        if arr.ndim == 2:
            arr = arr[None]
    ok = np.isfinite(arr).all(axis=0)
    if nodata is not None and np.isfinite(nodata):
        ok &= ~(arr == nodata).all(axis=0)
    return ok


def rescale_bands(image: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Stretch every band to 0-255 using percentile clipping.

    Segmentation parameters such as the multiresolution *scale* only mean
    the same thing across images if the radiometry is on a common range,
    so the segmenters call this by default. The percentiles are taken over the
    finite pixels of each band, and pixels that are not finite stay NaN.
    """
    out = np.empty(image.shape, dtype=np.float64)
    for b in range(image.shape[0]):
        x = image[b].astype(np.float64)
        ok = np.isfinite(x)
        if not ok.any():
            out[b] = np.nan
            continue
        lo, hi = np.percentile(x[ok], [low, high])
        if hi <= lo:
            out[b] = np.where(ok, 0.0, np.nan)
        else:
            out[b] = np.where(ok, np.clip((x - lo) / (hi - lo), 0, 1) * 255.0, np.nan)
    return out


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Rescaled display bands (NaN allowed) -> uint8, with invalid pixels shown as black."""
    return np.nan_to_num(rgb, nan=0.0).astype(np.uint8)


def save_raster(path: str, array: np.ndarray, transform=None, crs=None,
                band_names: list[str] | None = None, dtype: str | None = None) -> str:
    """Write a 2-D (rows, cols) or 3-D (bands, rows, cols) array as a GeoTIFF."""
    import rasterio
    from affine import Affine
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype or str(arr.dtype), crs=crs,
                       transform=transform or Affine.identity()) as dst:
        dst.write(arr.astype(dtype or arr.dtype))
        if band_names:
            for i, n in enumerate(band_names, start=1):
                dst.set_band_description(i, str(n))
    return path
