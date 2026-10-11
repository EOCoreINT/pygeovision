"""Regression tests for the ESA WorldCover / land-cover labelers (offline: tiles are local GeoTIFFs).

Real defects fixed:
  1. Each tile was written with ``from_bounds(*bbox, ...)`` even when the bbox window was clipped by the
     tile edge, so a bbox spanning two tiles stretched each piece over the whole bbox (mis-georeferenced).
  2. ``remap_classes`` was applied sequentially in place: {10: 20, 20: 30} turned every 10 into 30.
  3. A failed download left a missing tile path in the list; the failure surfaced as ``success: False``
     with no ``error``.
  4. The "Dynamic World" Planetary Computer backend downloaded the **entire** annual tile (not the bbox),
     picked an arbitrary item, ignored ``date_range`` and ``label_mode``, and reported Dynamic World's
     class map although the file uses the Impact Observatory numbering (1 water ... 7 built ...).
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from pygeovision.labeling import landcover as lc  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, features):
        self._f = features

    def json(self):
        return {"features": self._f}


def _tile(path, bounds, value, shape=(100, 100), crs="EPSG:4326", dtype="uint8"):
    from rasterio.transform import from_bounds
    arr = np.full(shape, value, dtype) if np.isscalar(value) else value
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs=crs, transform=from_bounds(*bounds, arr.shape[1], arr.shape[0])) as d:
        d.write(arr[np.newaxis])
    return str(path)


@pytest.fixture
def stac(monkeypatch):
    import requests

    def install(features):
        monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: _Resp(features))
    return install


def test_bbox_spanning_two_tiles_is_georeferenced_correctly(tmp_path, stac):
    a = _tile(tmp_path / "a.tif", (10.0, 50.0, 10.1, 50.1), 10)
    b = _tile(tmp_path / "b.tif", (10.1, 50.0, 10.2, 50.1), 50)
    stac([{"id": "a", "assets": {"map": {"href": a}}}, {"id": "b", "assets": {"map": {"href": b}}}])
    out = tmp_path / "lc.tif"
    res = lc.ESAWorldCoverLabeler(2021).label((10.05, 50.0, 10.15, 50.05), output_path=out)
    assert res["success"], res
    assert res["n_tiles"] == 2
    with rasterio.open(out) as src:
        arr = src.read(1)
        # the output must cover the bbox (within a pixel), not a stretched copy of it
        assert abs(src.bounds.left - 10.05) < 0.002 and abs(src.bounds.right - 10.15) < 0.002
        left = arr[:, : arr.shape[1] // 2 - 3]
        right = arr[:, arr.shape[1] // 2 + 3:]
        assert set(np.unique(left)) == {10} and set(np.unique(right)) == {50}
        # the boundary between the classes sits at lon 10.1
        col = src.index(10.1, 50.02)[1]
        assert abs(int(np.argmax(arr[arr.shape[0] // 2] == 50)) - col) <= 1
    assert set(res["classes_found"]) == {10, 50}


def test_remap_does_not_chain(tmp_path, stac):
    data = np.zeros((20, 20), "uint8")
    data[:, :10] = 10
    data[:, 10:] = 20
    t = _tile(tmp_path / "t.tif", (10.0, 50.0, 10.1, 50.1), data)
    stac([{"id": "t", "assets": {"map": {"href": t}}}])
    res = lc.ESAWorldCoverLabeler(2021).label((10.01, 50.01, 10.09, 50.09), output_path=tmp_path / "o.tif",
                                              remap_classes={10: 20, 20: 30})
    assert res["success"]
    assert set(res["classes_found"]) == {20, 30}, "10 was remapped to 20 and then on to 30"


def test_reference_raster_alignment(tmp_path, stac):
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    t = _tile(tmp_path / "t.tif", (10.0, 50.0, 10.1, 50.1), 40)
    stac([{"id": "t", "assets": {"map": {"href": t}}}])
    (x0,), (y0,) = warp_transform("EPSG:4326", "EPSG:32632", [10.02], [50.06])
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=64, width=80, count=1, dtype="uint8", crs="EPSG:32632",
                       transform=from_origin(x0, y0, 10, 10)) as d:
        d.write(np.zeros((1, 64, 80), "uint8"))
    out = tmp_path / "o.tif"
    res = lc.ESAWorldCoverLabeler(2021).label((10.0, 50.0, 10.1, 50.1), output_path=out, reference_raster=str(ref))
    assert res["success"]
    with rasterio.open(out) as src, rasterio.open(ref) as r:
        assert src.crs == r.crs and src.transform == r.transform and src.shape == r.shape
        assert set(np.unique(src.read(1))) == {40}


def test_unreadable_tile_gives_an_error_message(tmp_path, stac):
    stac([{"id": "x", "assets": {"map": {"href": str(tmp_path / "missing.tif")}}}])
    res = lc.ESAWorldCoverLabeler(2021).label((10.0, 50.0, 10.1, 50.1), output_path=tmp_path / "o.tif")
    assert res["success"] is False and "could not read" in res["error"]


def test_no_tiles_and_http_errors_explain_themselves(tmp_path, monkeypatch, stac):
    stac([])
    r = lc.ESAWorldCoverLabeler(2021).label((10.0, 50.0, 10.1, 50.1), output_path=tmp_path / "o.tif")
    assert not r["success"] and "No ESA WorldCover 2021 tiles" in r["error"]
    import requests

    class Bad:
        status_code = 503
    monkeypatch.setattr(requests, "post", lambda *a, **k: Bad())
    r = lc.ESAWorldCoverLabeler(2021).label((10.0, 50.0, 10.1, 50.1), output_path=tmp_path / "o.tif")
    assert not r["success"] and "503" in r["error"]


def test_pc_backend_reports_the_io_class_map_and_picks_the_newest_year(tmp_path, stac):
    old = _tile(tmp_path / "y22.tif", (10.0, 50.0, 10.1, 50.1), 2)
    new = _tile(tmp_path / "y23.tif", (10.0, 50.0, 10.1, 50.1), 7)
    stac([{"id": "a", "properties": {"start_datetime": "2022-01-01T00:00:00Z"}, "assets": {"data": {"href": old}}},
          {"id": "b", "properties": {"start_datetime": "2023-01-01T00:00:00Z"}, "assets": {"data": {"href": new}}}])
    out = tmp_path / "o.tif"
    res = lc.DynamicWorldLabeler().label((10.02, 50.02, 10.05, 50.05), ("2022-01-01", "2023-12-31"), out)
    assert res["success"] and res["year"] == 2023
    assert set(res["classes_found"]) == {7}
    assert res["class_map"][7] == "built" and 6 not in res["class_map"]       # IO numbering, not DW's
    assert res["class_map"] is lc.IO_LULC_CLASSES
    assert "NOT Google Dynamic World" in res["product"]
    with rasterio.open(out) as src:                 # clipped to the bbox, not the whole 0.1 deg tile
        assert src.width < 60 and src.height < 60


def test_pc_backend_remap_to_dynamic_world_numbers(tmp_path, stac):
    t = _tile(tmp_path / "t.tif", (10.0, 50.0, 10.1, 50.1), 7)
    stac([{"id": "a", "properties": {"start_datetime": "2023-01-01T00:00:00Z"}, "assets": {"data": {"href": t}}}])
    res = lc.DynamicWorldLabeler().label((10.02, 50.02, 10.05, 50.05), output_path=tmp_path / "o.tif",
                                         remap_classes=lc.IO_TO_DYNAMIC_WORLD)
    assert set(res["classes_found"]) == {6} and res["class_map"] is None


def test_pc_backend_no_items_names_the_year_limit(tmp_path, stac):
    stac([])
    res = lc.DynamicWorldLabeler().label((10.0, 50.0, 10.1, 50.1), ("2025-01-01", "2025-12-31"), tmp_path / "o.tif")
    assert not res["success"] and "2017-2023" in res["error"]


def test_unsupported_label_mode_and_backend(tmp_path):
    r = lc.DynamicWorldLabeler().label((0, 0, 1, 1), output_path=tmp_path / "o.tif", label_mode="all_probabilities")
    assert not r["success"] and "not supported" in r["error"]
    r = lc.DynamicWorldLabeler(backend="nope").label((0, 0, 1, 1), output_path=tmp_path / "o.tif")
    assert not r["success"] and "Unknown backend" in r["error"]


def test_earth_engine_backend_is_honest_without_the_package(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ee", None)
    r = lc.DynamicWorldLabeler(backend="earth_engine").label((0, 0, 1, 1), output_path=tmp_path / "o.tif")
    assert not r["success"] and "earthengine-api" in r["error"]


def test_align_to_reference_never_blends_classes(tmp_path):
    from rasterio.transform import from_origin
    data = np.zeros((40, 40), "uint8")
    data[:, :20] = 10
    data[:, 20:] = 80
    src = _tile(tmp_path / "s.tif", (0.0, 0.0, 0.04, 0.04), data)
    ref = tmp_path / "r.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=17, width=23, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(0.0, 0.04, 0.04 / 23, 0.04 / 17)) as d:
        d.write(np.zeros((1, 17, 23), "uint8"))
    out = lc.align_to_reference(src, ref, tmp_path / "a.tif")
    with rasterio.open(out) as o:
        assert set(np.unique(o.read(1))) <= {10, 80}
