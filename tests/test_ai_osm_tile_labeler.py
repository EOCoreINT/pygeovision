"""Regression tests for the tile-oriented OSM labeler (``pygeovision.ai.labeling.OSMLabeler``).

Real defects fixed:
  1. Roads were closed into polygons, so a U-shaped road became a filled blob.
  2. Every way the query returned was painted -- including untagged member ways of relations and ways
     of other kinds -- and anything unrecognised silently became class 1.
  3. ``feature_type="all"`` used the land-use table, so buildings, roads and water all came out as class 1.
  4. Lakes (``multipolygon`` relations) were never assembled.
  5. The mask was burned on a lon/lat envelope of the tile instead of on the tile's own grid, which skews
     labels on projected (UTM) tiles.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("shapely", reason="shapely not installed")

from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from pygeovision.ai.data.dataset import TileMetadata  # noqa: E402
from pygeovision.ai.labeling.osm_labeler import OSMLabeler  # noqa: E402
from pygeovision.core.exceptions import OSMLabelingError  # noqa: E402


def _way(i, tags, pts):
    return {"type": "way", "id": i, "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _utm_tile(tmp_path, size=100, res=10.0):
    (x0,), (y0,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0], [50.01])
    tf = from_origin(x0, y0, res, res)
    return TileMetadata(
        tile_id="t0", source_file=tmp_path / "s.tif", bounds=(x0, y0 - size * res, x0 + size * res, y0),
        crs="EPSG:32632", transform=list(tf)[:6], row_off=0, col_off=0, height=size, width=size,
        bands=[1], resolution_m=res, overlap=0, provider="x", satellite="x", datetime="", cloud_cover=0.0, extra={},
    )


def _labeler(monkeypatch, elements, feature_type):
    lab = OSMLabeler(feature_type=feature_type, request_delay=0.0)
    from pygeovision.labeling import osm as area

    monkeypatch.setattr(area.OSMLabeler, "_fetch_overpass", lambda self, q: {"elements": elements})
    return lab


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src


def test_road_is_a_line_not_a_blob(tmp_path, monkeypatch):
    road = _way(1, {"highway": "secondary"}, [(10.001, 50.001), (10.001, 50.009), (10.009, 50.009)])
    res = _labeler(monkeypatch, [road], "roads").label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.success, res.error
    arr, src = _read(res.label_path)
    h, w = arr.shape
    assert str(src.crs).endswith("32632") and arr.shape == (100, 100)
    assert arr[h // 2, w // 2] == 0, "the inside of the L was filled"
    assert 30 < int((arr == 1).sum()) < 1500


def test_all_gives_distinct_classes(tmp_path, monkeypatch):
    els = [_way(1, {"building": "yes"}, _rect(10.002, 50.002, 10.004, 50.004)),
           _way(2, {"natural": "water"}, _rect(10.006, 50.006, 10.008, 50.008)),
           _way(3, {"landuse": "residential"}, _rect(10.001, 50.001, 10.009, 50.009))]
    res = _labeler(monkeypatch, els, "all").label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.success, res.error
    arr, _ = _read(res.label_path)
    assert {1, 3, 7} <= set(np.unique(arr).tolist())
    assert (arr == 1).sum() > 0 and (arr == 3).sum() > 0 and (arr == 7).sum() > (arr == 1).sum()


def test_unrecognised_ways_are_not_painted(tmp_path, monkeypatch):
    els = [_way(1, {"building": "yes"}, _rect(10.002, 50.002, 10.004, 50.004)),
           _way(2, {}, _rect(10.006, 50.006, 10.008, 50.008)),                       # untagged member way
           _way(3, {"barrier": "fence"}, _rect(10.001, 50.007, 10.003, 50.009))]
    res = _labeler(monkeypatch, els, "buildings").label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.success, res.error
    arr, _ = _read(res.label_path)
    assert set(np.unique(arr).tolist()) == {0, 1}
    ys, xs = np.nonzero(arr == 1)
    # only the one real building: its extent is ~0.002 deg wide, not the union with the other ways
    assert xs.max() - xs.min() < 40


def test_lake_relation_with_island(tmp_path, monkeypatch):
    outer, inner = _rect(10.001, 50.001, 10.009, 50.009), _rect(10.004, 50.004, 10.006, 50.006)
    rel = {"type": "relation", "id": 9, "tags": {"natural": "water", "type": "multipolygon"},
           "members": [{"type": "way", "role": "outer", "geometry": [{"lon": x, "lat": y} for x, y in outer]},
                       {"type": "way", "role": "inner", "geometry": [{"lon": x, "lat": y} for x, y in inner]}]}
    res = _labeler(monkeypatch, [rel], "water").label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.success, res.error
    arr, _ = _read(res.label_path)
    (ix,), (iy,) = warp_transform("EPSG:4326", "EPSG:32632", [10.005], [50.005])
    r, c = rasterio.open(res.label_path).index(ix, iy)
    assert arr[r, c] == 0, "island not cut out"
    assert arr[20, 20] == 1


def test_building_lands_where_the_tile_grid_says(tmp_path, monkeypatch):
    house = _way(1, {"building": "yes"}, _rect(10.003, 50.004, 10.004, 50.005))
    tile = _utm_tile(tmp_path)
    res = _labeler(monkeypatch, [house], "buildings").label_tile(tile, tmp_path, "segmentation")
    arr, src = _read(res.label_path)
    (cx,), (cy,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0035], [50.0045])
    r, c = src.index(cx, cy)
    assert arr[r, c] == 1


def test_empty_area_is_skipped_not_labelled(tmp_path, monkeypatch):
    res = _labeler(monkeypatch, [], "buildings").label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.skipped and not res.success


def test_failed_request_is_an_error(tmp_path, monkeypatch):
    from pygeovision.labeling import osm as area

    monkeypatch.setattr(area.OSMLabeler, "_fetch_overpass", lambda self, q: None)
    res = OSMLabeler(request_delay=0.0).label_tile(_utm_tile(tmp_path), tmp_path, "segmentation")
    assert res.error and "Overpass" in res.error and not res.skipped


def test_class_names_match_values():
    assert OSMLabeler._get_class_names("buildings") == ["background", "building"]
    assert OSMLabeler._get_class_names("roads") == ["background", "road"]
    names = OSMLabeler._get_class_names("all")
    assert names[1] == "buildings" and names[7] == "residential" and len(names) == 11
    assert OSMLabeler._get_class_names("landuse") == ["background", "residential", "commercial", "agricultural", "vegetation"]


def test_unknown_feature_type_raises():
    with pytest.raises(OSMLabelingError):
        OSMLabeler(feature_type="rivers")
