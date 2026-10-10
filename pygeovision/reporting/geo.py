"""Raster measurements for reports: areas, class tables, statistics, binning.

Everything here reads the pipeline's own output files, so the numbers in a
report are measured from the delivered rasters (and can be compared with the
statistics the pipeline itself returned).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.reporting.palettes import ClassDef, categorical
from pygeovision.reporting.specs import Binning

logger = logging.getLogger(__name__)

_WGS84_A = 6_378_137.0
_WGS84_E2 = 0.00669437999014
_MAX_STAT_PIXELS = 40_000_000     # above this, continuous statistics use a decimated read


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Geodesic area of a lon/lat box in km2."""
    from pyproj import Geod
    w, s, e, n = bbox
    area, _ = Geod(ellps="WGS84").polygon_area_perimeter([w, e, e, w], [s, s, n, n])
    return abs(area) / 1e6


def _q(lat_rad: np.ndarray) -> np.ndarray:
    """Authalic function of the WGS84 ellipsoid: the area of a lat band is proportional to q(lat2) - q(lat1)."""
    e = math.sqrt(_WGS84_E2)
    s = np.sin(lat_rad)
    return s / (1 - _WGS84_E2 * s * s) + np.log((1 + e * s) / (1 - e * s)) / (2 * e)


def ellipsoid_row_areas_ha(top: float, cell_lat_deg: float, cell_lon_deg: float, n_rows: int) -> np.ndarray:
    """Area in ha of one cell in each of ``n_rows`` rows of a lon/lat grid (exact on the WGS84 ellipsoid)."""
    edges = np.radians(top - cell_lat_deg * np.arange(n_rows + 1))
    q = _q(edges)
    m2 = _WGS84_A ** 2 * (1 - _WGS84_E2) / 2.0 * math.radians(cell_lon_deg) * np.abs(q[:-1] - q[1:])
    return m2 / 1e4


def _is_geographic(crs: Any) -> bool:
    return crs is None or bool(getattr(crs, "is_geographic", False))


def row_pixel_area_ha(src: Any) -> np.ndarray:
    """Area of one pixel in each raster row, in hectares.

    Geographic rasters have rows of different area (a degree of longitude
    shrinks towards the poles); projected rasters use the declared linear unit.
    """
    a, e = abs(src.transform.a), abs(src.transform.e)
    h = src.height
    if _is_geographic(src.crs):
        return ellipsoid_row_areas_ha(src.bounds.top, e, a, h)
    factor = float(src.crs.linear_units_factor[1]) if src.crs is not None else 1.0
    return np.full(h, a * e * factor * factor / 1e4)


@dataclass
class ClassTable:
    rows: list[dict[str, Any]]            # value, name, color, pixels, area_ha, share (of valid), share_aoi
    valid_pixels: int
    total_pixels: int
    valid_area_ha: float
    total_area_ha: float


def class_table(path: Path, band: int, defs: dict[int, ClassDef], *, extra_nodata: tuple[int, ...] = ()) -> ClassTable:
    """Pixel counts and areas per class value, over valid pixels."""
    import rasterio

    areas: dict[int, float] = {}
    counts: dict[int, int] = {}
    total_px = 0
    with rasterio.open(path) as src:
        row_ha = row_pixel_area_ha(src)
        nodata = src.nodata
        total_area = float(row_ha.sum() * src.width)
        for _, win in src.block_windows(band):
            arr = src.read(band, window=win)
            r0 = int(win.row_off)
            w = row_ha[r0:r0 + arr.shape[0]][:, None]
            total_px += arr.size
            valid = np.ones(arr.shape, dtype=bool)
            if np.issubdtype(arr.dtype, np.floating):
                valid &= np.isfinite(arr)
            if nodata is not None and np.isfinite(nodata):
                valid &= arr != nodata
            for nd in extra_nodata:
                valid &= arr != nd
            if not valid.any():
                continue
            vals, inv = np.unique(arr[valid].astype(np.int64), return_inverse=True)
            wts = np.broadcast_to(w, arr.shape)[valid]
            a_sum = np.bincount(inv, weights=wts, minlength=len(vals))
            c_sum = np.bincount(inv, minlength=len(vals))
            for v, a_, c_ in zip(vals.tolist(), a_sum.tolist(), c_sum.tolist()):
                areas[v] = areas.get(v, 0.0) + a_
                counts[v] = counts.get(v, 0) + int(c_)
    valid_px = sum(counts.values())
    valid_area = sum(areas.values())
    rows = []
    for v in sorted(set(areas) | {d for d in defs if d in counts}):
        d = defs.get(v) or ClassDef(v, f"Class {v}", categorical(v))
        rows.append({
            "value": v, "name": d.name, "color": d.color, "pixels": counts.get(v, 0), "area_ha": areas.get(v, 0.0),
            "share": (areas.get(v, 0.0) / valid_area) if valid_area > 0 else None,
            "share_aoi": (areas.get(v, 0.0) / total_area) if total_area > 0 else None,
        })
    return ClassTable(rows, valid_px, total_px, valid_area, total_area)


def read_band(path: Path, band: int, *, max_pixels: int = _MAX_STAT_PIXELS) -> tuple[np.ndarray, Any, bool]:
    """Band as float32 with nodata -> NaN, plus the dataset's georeferencing info.

    Returns ``(array, profile_dict, decimated)``; ``decimated`` is True if the raster was
    larger than ``max_pixels`` and was read at reduced resolution (statistics stay representative).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        decimated = h * w > max_pixels
        if decimated:
            f = math.sqrt(max_pixels / (h * w))
            shape = (max(1, int(h * f)), max(1, int(w * f)))
            arr = src.read(band, out_shape=shape, resampling=Resampling.nearest).astype(np.float32)
        else:
            arr = src.read(band).astype(np.float32)
        nodata = src.nodata
        info = {"crs": src.crs, "bounds": src.bounds, "transform": src.transform, "width": w, "height": h,
                "dtype": src.dtypes[band - 1], "nodata": nodata}
    if nodata is not None and np.isfinite(nodata):
        arr[arr == np.float32(nodata)] = np.nan
    return arr, info, decimated


def continuous_summary(arr: np.ndarray) -> dict[str, Any]:
    """Descriptive statistics over the finite pixels of ``arr``."""
    ok = np.isfinite(arr)
    n_valid = int(ok.sum())
    out: dict[str, Any] = {"n_valid": n_valid, "n_total": int(arr.size),
                           "valid_fraction": (n_valid / arr.size) if arr.size else None}
    if n_valid == 0:
        return out
    v = arr[ok].astype(np.float64)
    p = np.percentile(v, [2, 5, 25, 50, 75, 95, 98])
    out.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()),
               p2=float(p[0]), p5=float(p[1]), p25=float(p[2]), median=float(p[3]), p75=float(p[4]),
               p95=float(p[5]), p98=float(p[6]))
    return out


def _sample_cmap(name: str, n: int) -> list[str]:
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    cm = colormaps[name]
    return [to_hex(cm(0.15 + 0.75 * i / max(1, n - 1))) for i in range(n)]


def apply_binning(arr: np.ndarray, b: Binning) -> tuple[np.ndarray, list[ClassDef]]:
    """Classify ``arr`` by ``b``. Returns uint8 classes (255 = invalid) and the class definitions."""
    ok = np.isfinite(arr)
    if not ok.any():
        return np.full(arr.shape, 255, np.uint8), []
    v = arr[ok].astype(np.float64)
    if b.kind == "fixed":
        edges = list(b.edges)
        names = list(b.names)
        colors = list(b.colors) or _sample_cmap(b.cmap, len(edges) + 1)
    elif b.kind == "equal":
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-9
        edges = list(np.linspace(lo, hi, b.n + 1)[1:-1])
        bounds = [lo] + edges + [hi]
        step = (hi - lo) / b.n
        nd = int(min(4, max(1, math.ceil(-math.log10(step)) + 1))) if step > 0 else 1
        u = f" {b.unit}" if b.unit else ""
        names = [f"{bounds[i]:.{nd}f} to {bounds[i + 1]:.{nd}f}{u}" for i in range(b.n)]
        names[0] = f"below {bounds[1]:.{nd}f}{u}"
        names[-1] = f"above {bounds[-2]:.{nd}f}{u}"
        colors = list(b.colors) or _sample_cmap(b.cmap, b.n)
    elif b.kind == "quantile":
        edges = sorted(set(float(x) for x in np.percentile(v, np.linspace(0, 100, b.n + 1)[1:-1])))
        names = list(b.names) if len(b.names) == len(edges) + 1 else [f"Part {i + 1} of {len(edges) + 1}" for i in range(len(edges) + 1)]
        colors = list(b.colors) or _sample_cmap(b.cmap, len(edges) + 1)
    elif b.kind == "std":
        m, s = float(v.mean()), float(v.std())
        edges = [m - b.k * s, m + b.k * s]
        names = list(b.names) or ["Low", "Mid", "High"]
        colors = list(b.colors) or _sample_cmap(b.cmap, 3)
    else:
        raise ValueError(f"unknown binning kind {b.kind!r}")
    cls = np.full(arr.shape, 255, np.uint8)
    cls[ok] = np.digitize(arr[ok], edges).astype(np.uint8)
    defs = [ClassDef(i, names[i], colors[i % len(colors)]) for i in range(len(edges) + 1)]
    return cls, defs


def table_from_classes(cls: np.ndarray, defs: list[ClassDef], info: dict[str, Any]) -> ClassTable:
    """Class table for an in-memory class array (used for binned continuous layers)."""
    h, w = cls.shape
    if _is_geographic(info["crs"]):
        a = abs(info["transform"].a) * info["width"] / w          # decimated reads have a coarser cell
        e = abs(info["transform"].e) * info["height"] / h
        row_ha = ellipsoid_row_areas_ha(info["bounds"].top, e, a, h)
    else:
        row_ha = np.full(h, abs(info["transform"].a) * abs(info["transform"].e) * (info["width"] / w) * (info["height"] / h) / 1e4)
    valid = cls != 255
    total_area = float(row_ha.sum() * w)
    valid_area = float((row_ha[:, None] * valid).sum())
    rows = []
    for d in defs:
        m = cls == d.value
        area = float((row_ha[:, None] * m).sum())
        rows.append({"value": d.value, "name": d.name, "color": d.color, "pixels": int(m.sum()), "area_ha": area,
                     "share": area / valid_area if valid_area > 0 else None,
                     "share_aoi": area / total_area if total_area > 0 else None})
    return ClassTable(rows, int(valid.sum()), int(cls.size), valid_area, total_area)


def aoi_extent_wgs84(info: dict[str, Any]) -> tuple[float, float, float, float]:
    """Raster bounds as (west, south, east, north) in degrees."""
    b = info["bounds"]
    if _is_geographic(info["crs"]):
        return (b.left, b.bottom, b.right, b.top)
    from rasterio.warp import transform_bounds
    return tuple(transform_bounds(info["crs"], "EPSG:4326", b.left, b.bottom, b.right, b.top))
