"""Regression tests for the OSM labeler's geometry handling.

Real defects fixed:
  1. Roads and rivers were closed into polygons (first node joined to last), so a U-shaped road
     was painted as a filled blob.
  2. Land-use polygons (value 7) were burned *after* buildings (value 1) because features were
     sorted by class value, so a residential zone erased every building inside it.
  3. With a ``reference_raster`` in a projected CRS the lon/lat geometry was burned with a UTM
     transform, so the label raster came out empty.
  4. ``multipolygon`` relations (big lakes, courtyard buildings) were dropped.
  5. The query asked for ``natural=wood|scrub|heath``, ``landuse=vineyard``, ``sports_centre`` etc.,
     but the converter only recognised a subset, so those features were downloaded and discarded.
  6. The default vector path equalled the raster path for outputs not ending in ``.tif``.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("shapely", reason="shapely not installed")

from pygeovision.labeling.osm import OSM_CATEGORIES, OSMLabeler, classify_tags  # noqa: E402

BBOX = (10.0, 50.0, 10.01, 50.01)


def _way(i, tags, pts):
    return {"type": "way", "id": i, "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1), src


def test_classification_matches_the_query():
    assert classify_tags({"natural": "wood"}) == ("vegetation", "area")
    assert classify_tags({"landuse": "vineyard"}) == ("vegetation", "area")
    assert classify_tags({"landuse": "greenhouse_horticulture"}) == ("agricultural", "area")
    assert classify_tags({"leisure": "sports_centre"}) == ("sports", "area")
    assert classify_tags({"highway": "primary_link"}) == ("roads", "line")
    assert classify_tags({"highway": "footway"}) is None
    assert classify_tags({"waterway": "river"}) == ("water", "line")
    assert classify_tags({"waterway": "riverbank"}) == ("water", "area")
    # a building that is also tagged as parking is a building
    assert classify_tags({"building": "yes", "amenity": "parking"})[0] == "buildings"
    assert classify_tags({"building": "no"}) is None


def test_every_category_has_a_recognisable_tag():
    sample = {"buildings": {"building": "yes"}, "roads": {"highway": "service"}, "water": {"natural": "water"},
              "vegetation": {"natural": "scrub"}, "agricultural": {"landuse": "farmland"},
              "commercial": {"landuse": "retail"}, "residential": {"landuse": "residential"},
              "solar_panels": {"generator:source": "solar"}, "parking": {"amenity": "parking"},
              "sports": {"leisure": "pitch"}}
    assert set(sample) == set(OSM_CATEGORIES)
    for cat, tags in sample.items():
        assert classify_tags(tags)[0] == cat


def test_road_is_a_buffered_line_not_a_filled_blob(tmp_path):
    # U-shaped road: closing it into a polygon would fill the whole interior
    road = _way(1, {"highway": "secondary"},
                [(10.001, 50.001), (10.001, 50.009), (10.009, 50.009), (10.009, 50.001)])
    lab = OSMLabeler()
    gj = lab._osm_to_geojson({"elements": [road]}, ["roads"])
    assert gj["features"][0]["geometry"]["type"] == "LineString"
    assert gj["features"][0]["properties"]["width_m"] == 8.0
    out = tmp_path / "roads.tif"
    lab._rasterise(gj, BBOX, out, resolution_m=10.0)
    arr, _ = _read(out)
    h, w = arr.shape
    assert arr[h // 2, w // 2] == 0, "the inside of the U was filled"
    n_road = int((arr == 2).sum())
    # the road is ~2.4 km long; at 10 m resolution that is a few hundred pixels, a filled U would be >3000
    assert 150 < n_road < 1500


def test_buildings_survive_inside_a_residential_zone(tmp_path):
    zone = _way(1, {"landuse": "residential"}, _rect(10.001, 50.001, 10.009, 50.009))
    house = _way(2, {"building": "yes"}, _rect(10.004, 50.004, 10.005, 50.005))
    lab = OSMLabeler()
    gj = lab._osm_to_geojson({"elements": [house, zone]}, list(OSM_CATEGORIES))
    out = tmp_path / "x.tif"
    lab._rasterise(gj, BBOX, out, resolution_m=10.0)
    arr, _ = _read(out)
    assert (arr == 1).sum() > 20, "the residential zone erased the building"
    assert (arr == 7).sum() > (arr == 1).sum()


def test_wood_inside_a_residential_zone_stays_wood(tmp_path):
    zone = _way(1, {"landuse": "residential"}, _rect(10.001, 50.001, 10.009, 50.009))
    wood = _way(2, {"natural": "wood"}, _rect(10.004, 50.004, 10.006, 50.006))
    lab = OSMLabeler()
    gj = lab._osm_to_geojson({"elements": [zone, wood]}, list(OSM_CATEGORIES))
    out = tmp_path / "x.tif"
    lab._rasterise(gj, BBOX, out, resolution_m=10.0)
    arr, _ = _read(out)
    h, w = arr.shape
    assert arr[h // 2, w // 2] == 4 and (arr == 7).sum() > (arr == 4).sum()


def test_multipolygon_relation_with_island(tmp_path):
    outer = _rect(10.001, 50.001, 10.009, 50.009)
    inner = _rect(10.004, 50.004, 10.006, 50.006)
    rel = {"type": "relation", "id": 9, "tags": {"natural": "water", "type": "multipolygon"},
           "members": [
               {"type": "way", "role": "outer", "geometry": [{"lon": x, "lat": y} for x, y in outer]},
               {"type": "way", "role": "inner", "geometry": [{"lon": x, "lat": y} for x, y in inner]},
           ]}
    lab = OSMLabeler()
    gj = lab._osm_to_geojson({"elements": [rel]}, ["water"])
    assert len(gj["features"]) == 1
    out = tmp_path / "lake.tif"
    lab._rasterise(gj, BBOX, out, resolution_m=10.0)
    arr, _ = _read(out)
    h, w = arr.shape
    assert arr[h // 2, w // 2] == 0, "island not cut out"
    assert arr[int(h * 0.2), int(w * 0.2)] == 3


def test_open_ring_area_is_skipped_and_counted():
    bad = _way(1, {"building": "yes"}, [(10.0, 50.0), (10.001, 50.0), (10.001, 50.001), (10.0, 50.001)])
    gj = OSMLabeler()._osm_to_geojson({"elements": [bad]}, ["buildings"])
    assert gj["features"] == [] and gj["n_skipped"] == 1


def test_reference_raster_in_utm_gets_correct_labels(tmp_path):
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    # 100 x 100 px at 10 m in UTM 32N around lon 10, lat 50
    (x0,), (y0,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0], [50.01])
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(x0, y0, 10, 10)) as dst:
        dst.write(np.zeros((1, 100, 100), "uint8"))
    house = _way(1, {"building": "yes"}, _rect(10.003, 50.004, 10.004, 50.005))
    lab = OSMLabeler()
    gj = lab._osm_to_geojson({"elements": [house]}, ["buildings"])
    out = tmp_path / "lab.tif"
    lab._rasterise(gj, BBOX, out, reference_raster=str(ref))
    arr, src = _read(out)
    assert str(src.crs).endswith("32632")
    assert arr.shape == (100, 100)
    assert (arr == 1).sum() > 20, "labels came out empty on a projected reference grid"
    # the building centre must sit at the pixel the CRS conversion predicts
    (cx,), (cy,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0035], [50.0045])
    r, c = src.index(cx, cy)
    assert arr[r, c] == 1


def test_label_end_to_end_with_fake_overpass(tmp_path, monkeypatch):
    house = _way(2, {"building": "yes"}, _rect(10.004, 50.004, 10.005, 50.005))
    lab = OSMLabeler()
    monkeypatch.setattr(lab, "_fetch_overpass", lambda q: {"elements": [house]})
    out = tmp_path / "labels.png"           # not a .tif: the vector file must not overwrite it
    res = lab.label(BBOX, ["buildings"], output_path=out)
    assert res["success"] and res["n_features"] == 1
    assert res["class_values"] == {"buildings": 1}
    assert res["vector_path"] != str(out)
    assert res["vector_path"].endswith("labels_vectors.geojson")
    arr, _ = _read(out)
    assert (arr == 1).sum() > 0


def test_overpass_runtime_error_is_a_failure(monkeypatch, tmp_path):
    lab = OSMLabeler()
    monkeypatch.setattr(lab, "_fetch_overpass",
                        lambda q: {"elements": [], "remark": "runtime error: Query timed out in 'query' at line 3"})
    res = lab.label(BBOX, ["buildings"], output_path=tmp_path / "x.tif")
    assert res["success"] is False and "timed out" in res["error"]


def test_failed_request_error_carries_the_http_status(monkeypatch, tmp_path):
    import requests

    class R:
        status_code = 429

    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    monkeypatch.setattr("time.sleep", lambda s: None)
    lab = OSMLabeler(retry_attempts=1)
    res = lab.label(BBOX, ["buildings"], output_path=tmp_path / "x.tif")
    assert res["success"] is False and "429" in res["error"]


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown OSM categories"):
        OSMLabeler().preview_query(BBOX, ["buildngs"])
    with pytest.raises(ValueError, match="Unknown OSM categories"):
        OSMLabeler().label(BBOX, ["roads", "rivers"])


def test_huge_grid_is_refused():
    with pytest.raises(ValueError, match="pixels"):
        OSMLabeler._grid_from_bbox((0.0, 0.0, 1.0, 1.0), 0.5)


def test_tiny_bbox_still_gets_one_pixel():
    _, w, h = OSMLabeler._grid_from_bbox((10.0, 50.0, 10.0000001, 50.0000001), 10.0)
    assert (w, h) == (1, 1)
