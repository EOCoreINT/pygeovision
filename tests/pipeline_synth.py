"""Synthetic rasters and a patched download layer for pipeline tests.

The pipelines' own logic (band maths, nodata handling, thresholds, file
writing, statistics) runs for real; only the network download is replaced
by pre-built GeoTIFFs whose correct answer is known by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.transform import from_origin

BBOX = (-0.15, 51.47, -0.10, 51.52)


def write_raster(path: Path, bands: Any, bounds=BBOX, crs: str = "EPSG:4326",
                 dtype: str = "float32", nodata: Any = None) -> Path:
    arr = np.asarray(bands, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[None]
    _, h, w = arr.shape
    transform = from_origin(bounds[0], bounds[3], (bounds[2] - bounds[0]) / w, (bounds[3] - bounds[1]) / h)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr)
    return path


def const(value: float, shape=(40, 40)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def patch_downloads(monkeypatch, single: Callable | dict | Path | None = None,
                    pair: tuple[Path, Path] | None = None, record: dict | None = None) -> None:
    """Replace the network download of every pipeline.

    ``single`` is a path, a ``{date: path}`` dict, or a callable
    ``(bbox, date, out, **kw) -> Path | None``. ``pair`` is (before, after).
    Every call's keyword arguments are appended to ``record["calls"]`` so a
    test can assert which bands the pipeline asked for.
    """
    from pygeovision.ai.pipelines import BasePipeline

    calls: list[dict] = record.setdefault("calls", []) if record is not None else []

    def fake_single(self, bbox, date, output_dir, **kw):
        calls.append({"date": date, **kw})
        if callable(single):
            return single(bbox, date, output_dir, **kw)
        if isinstance(single, dict):
            return single.get(date)
        return single

    def fake_pair(self, bbox, date_before, date_after, output_dir, **kw):
        calls.append({"date_before": date_before, "date_after": date_after, **kw})
        return pair if pair is not None else (None, None)

    monkeypatch.setattr(BasePipeline, "_search_and_download", fake_single)
    monkeypatch.setattr(BasePipeline, "_search_and_download_pair", fake_pair)


def no_network(monkeypatch) -> dict:
    """Make any download attempt fail the test (proves a pipeline failed fast)."""
    from pygeovision.ai.pipelines import BasePipeline

    hit = {"n": 0}

    def boom(self, *a, **k):
        hit["n"] += 1
        raise AssertionError("pipeline tried to download imagery")

    monkeypatch.setattr(BasePipeline, "_search_and_download", boom)
    monkeypatch.setattr(BasePipeline, "_search_and_download_pair", boom)
    return hit


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()
