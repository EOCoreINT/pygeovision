"""Regression tests for pygeovision.ai.labeling.dynamic_world.DynamicWorldLabeler.

Real defects fixed:
  1. When no scene was found it wrote an all-zero mask -- class 0 is *water* in Dynamic World -- and
     returned a result with no error, i.e. a "successful" tile labelled entirely water.
  2. Pixels not covered by a scene were left at 0 (= water) instead of no-data.
  3. ``composite_method="probability"`` / ``probability_bands=True`` were accepted and silently produced
     the most-frequent class.
"""
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler  # noqa: E402


def _tile(path, size=20):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint16",
                       crs="EPSG:32632", transform=from_origin(500000, 5500000, 10, 10)) as d:
        d.write(np.ones((3, size, size), "uint16"))
    return Path(path)


def _scene(path, arr, x0=500000, y0=5500000):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(x0, y0, 10, 10)) as d:
        d.write(arr[np.newaxis])
    return Path(path)


def test_no_scenes_is_an_error_and_writes_nothing(tmp_path, monkeypatch):
    lab = DynamicWorldLabeler(cache_dir=tmp_path / "c")
    monkeypatch.setattr(lab, "_fetch_dynamic_world_scenes", lambda bbox: [])
    out = tmp_path / "lab.tif"
    r = lab.label_tile(_tile(tmp_path / "t.tif"), None, out)
    assert not r.success and "No Dynamic World scenes" in r.error
    assert r.label_path is None and not out.exists()


def test_uncovered_pixels_are_nodata_not_water(tmp_path, monkeypatch):
    lab = DynamicWorldLabeler(cache_dir=tmp_path / "c")
    half = np.full((20, 10), 6, "uint8")                       # covers only the left half of the 20x20 tile
    scene = _scene(tmp_path / "s.tif", half)
    monkeypatch.setattr(lab, "_fetch_dynamic_world_scenes", lambda bbox: [scene])
    out = tmp_path / "lab.tif"
    r = lab.label_tile(_tile(tmp_path / "t.tif"), None, out)
    assert r.success
    with rasterio.open(out) as o:
        arr = o.read(1)
        assert o.nodata == 255
    assert (arr[:, :10] == 6).all() and (arr[:, 10:] == 255).all()


def test_mode_ignores_scenes_that_do_not_cover_a_pixel(tmp_path, monkeypatch):
    lab = DynamicWorldLabeler(cache_dir=tmp_path / "c")
    a = _scene(tmp_path / "a.tif", np.full((20, 20), 1, "uint8"))
    b = _scene(tmp_path / "b.tif", np.full((20, 20), 1, "uint8"))
    c = _scene(tmp_path / "c.tif", np.full((20, 10), 4, "uint8"))
    monkeypatch.setattr(lab, "_fetch_dynamic_world_scenes", lambda bbox: [a, b, c])
    out = tmp_path / "lab.tif"
    assert lab.label_tile(_tile(tmp_path / "t.tif"), None, out).success
    with rasterio.open(out) as o:
        assert (o.read(1) == 1).all()                          # 2 votes for trees beat 1 for crops


def test_probability_options_are_refused():
    with pytest.raises(NotImplementedError):
        DynamicWorldLabeler(composite_method="probability")
    with pytest.raises(NotImplementedError):
        DynamicWorldLabeler(probability_bands=True)
