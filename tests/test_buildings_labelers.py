"""Regression tests for the Microsoft / Google building labelers.

Real defects fixed:
  1. With a projected ``reference_raster`` the lon/lat footprints were burned with a UTM transform
     -> an empty label raster that still looked like a normal output.
  2. The bbox test looked only at a footprint's first vertex with a 0.1 deg margin, so buildings
     well outside the bbox were counted and written to the GeoJSON.
  3. Microsoft marks "confidence not available" as -1; with the default ``min_confidence=0.5`` every
     such footprint was dropped, so whole countries came back empty.
  4. A failed download and a genuinely empty area both produced the same unexplained 0.
  5. The vector file name was derived with ``str.replace(".tif", ...)``, which wrote the GeoJSON over
     the raster for other extensions.
"""
import gzip
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("shapely", reason="shapely not installed")
pd = pytest.importorskip("pandas")

from pygeovision.labeling.buildings import GoogleBuildingsLabeler, MicrosoftBuildingsLabeler  # noqa: E402

BBOX = (10.0, 50.0, 10.01, 50.01)


def _poly(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _ms_payload(records):
    lines = "\n".join(json.dumps({"type": "Feature", "geometry": g, "properties": p}) for g, p in records)
    return gzip.compress(lines.encode())


class _Resp:
    def __init__(self, content=b"", status=200):
        self.content, self.status_code = content, status


def _ms_lab(monkeypatch, records, status=200):
    import requests
    lab = MicrosoftBuildingsLabeler()
    monkeypatch.setattr(lab, "_load_manifest", lambda: pd.DataFrame({
        "Location": ["X"], "QuadKey": [lab._bbox_to_quadkeys(BBOX, 9)[0][:6]],
        "Url": ["https://example.test/a.csv.gz"], "Size": ["1"], "UploadDate": ["2026"]}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_ms_payload(records), status))
    return lab


def test_negative_confidence_means_unavailable_not_low(monkeypatch):
    lab = _ms_lab(monkeypatch, [(_poly(10.002, 50.002, 10.003, 50.003), {"confidence": -1.0}),
                                (_poly(10.004, 50.004, 10.005, 50.005), {"confidence": 0.2}),
                                (_poly(10.006, 50.006, 10.007, 50.007), {"confidence": 0.9})])
    gj = lab._fetch_buildings_geojson(BBOX)
    confs = sorted(f["properties"]["confidence"] for f in gj["features"])
    assert confs == [-1.0, 0.9]            # 0.2 < 0.5 dropped, -1 kept
    assert gj["n_no_confidence"] == 1


def test_footprints_outside_the_bbox_are_dropped(monkeypatch):
    lab = _ms_lab(monkeypatch, [(_poly(10.002, 50.002, 10.003, 50.003), {"confidence": 0.9}),
                                (_poly(10.05, 50.05, 10.051, 50.051), {"confidence": 0.9})])  # 5 km away
    assert len(lab._fetch_buildings_geojson(BBOX)["features"]) == 1


def test_microsoft_reference_raster_in_utm(monkeypatch, tmp_path):
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    (x0,), (y0,) = warp_transform("EPSG:4326", "EPSG:32632", [10.0], [50.01])
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(x0, y0, 10, 10)) as d:
        d.write(np.zeros((1, 100, 100), "uint8"))
    lab = _ms_lab(monkeypatch, [(_poly(10.003, 50.004, 10.004, 50.005), {"confidence": 0.9})])
    out = tmp_path / "ms.png"      # not .tif -> vector must not overwrite the raster
    res = lab.label(BBOX, output_path=out, reference_raster=str(ref))
    assert res["success"] and res["n_buildings"] == 1
    assert res["vector_path"] != str(out)
    with rasterio.open(out) as src:
        arr = src.read(1)
        assert str(src.crs).endswith("32632") and arr.shape == (100, 100)
    assert (arr == 1).sum() > 20, "labels came out empty on a projected reference grid"


def test_microsoft_failure_reasons_are_distinct(monkeypatch, tmp_path):
    # manifest unreachable
    lab = MicrosoftBuildingsLabeler()
    monkeypatch.setattr(lab, "_load_manifest", lambda: (_ for _ in ()).throw(RuntimeError("dns")))
    res = lab.label(BBOX, output_path=tmp_path / "a.tif")
    assert not res["success"] and "manifest" in res["error"]
    # files matched but every download fails
    lab = _ms_lab(monkeypatch, [], status=500)
    res = lab.label(BBOX, output_path=tmp_path / "b.tif")
    assert not res["success"] and "could be downloaded" in res["error"] and res["n_files_failed"] == 1
    # files read, nothing inside
    lab = _ms_lab(monkeypatch, [(_poly(11, 51, 11.001, 51.001), {"confidence": 0.9})])
    res = lab.label(BBOX, output_path=tmp_path / "c.tif")
    assert not res["success"] and "inside bbox" in res["error"]
    # no manifest rows cover the bbox
    lab = MicrosoftBuildingsLabeler()
    monkeypatch.setattr(lab, "_load_manifest", lambda: pd.DataFrame(
        {"Location": ["X"], "QuadKey": ["333333"], "Url": ["u"], "Size": ["1"], "UploadDate": ["1"]}))
    res = lab.label(BBOX, output_path=tmp_path / "d.tif")
    assert not res["success"] and "no coverage" in res["error"]


def _google_csv(rows):
    head = "latitude,longitude,area_in_meters,confidence,geometry,full_plus_code\n"
    body = "".join(f'{la},{lo},50.0,{c},"{wkt}",X\n' for la, lo, c, wkt in rows)
    return gzip.compress((head + body).encode())


def test_google_filters_by_bbox_and_confidence(monkeypatch, tmp_path):
    import requests
    in_wkt = "POLYGON((10.002 50.002, 10.003 50.002, 10.003 50.003, 10.002 50.003, 10.002 50.002))"
    far_wkt = "POLYGON((10.5 50.5, 10.501 50.5, 10.501 50.501, 10.5 50.501, 10.5 50.5))"
    payload = _google_csv([(50.0025, 10.0025, 0.9, in_wkt), (50.5, 10.5, 0.95, far_wkt),
                           (50.0025, 10.0025, 0.3, in_wkt)])
    lab = GoogleBuildingsLabeler(min_confidence=0.7)
    monkeypatch.setattr(lab, "_bbox_to_s2_cells", lambda b: ["abc"])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(payload))
    res = lab.label(BBOX, output_path=tmp_path / "g.tif")
    assert res["success"] and res["n_buildings"] == 1
    assert res["vector_path"].endswith("g_google_buildings.geojson")


def test_google_no_coverage_is_explained(monkeypatch, tmp_path):
    import requests
    lab = GoogleBuildingsLabeler()
    monkeypatch.setattr(lab, "_bbox_to_s2_cells", lambda b: ["abc", "def"])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(b"", 404))
    res = lab.label(BBOX, output_path=tmp_path / "g.tif")
    assert not res["success"] and "no coverage" in res["error"] and res["n_cells_failed"] == 2


def test_huge_default_grid_is_refused(monkeypatch, tmp_path):
    lab = _ms_lab(monkeypatch, [(_poly(0.1, 0.1, 0.2, 0.2), {"confidence": 0.9})])
    with pytest.raises(ValueError, match="pixels"):
        lab.label((0.0, 0.0, 1.0, 1.0), output_path=tmp_path / "x.tif", resolution_m=1.0)
