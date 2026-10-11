"""Regression tests for AutoLabelPipeline fusion.

Real defects fixed:
  1. ``scipy.stats.mode(...).mode[0]`` took the first *row* of the result in current SciPy, so
     ``majority_vote`` (the default) wrote a wrong-shaped array, failed, and the pipeline returned
     ``fused_path=None`` without saying so.
  2. Rasters of different shapes were nearest-resized to the first one's shape, ignoring their
     georeferencing, and the fused raster inherited the first source's profile (nodata, dtype).
  3. Rasters with different class alphabets (OSM 1=building, 2=road; ESA 10=tree, 50=built-up; SAM
     instance ids) were voted on as if comparable.
  4. An unknown ``fusion`` silently returned the first source; unknown sources failed only at run time.
  5. A source that only *submitted* an export (Earth Engine) counted as a succeeded label.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from pygeovision.labeling.pipeline import AutoLabelPipeline  # noqa: E402

BOUNDS = (10.0, 50.0, 10.1, 50.1)


def _raster(path, arr, bounds=BOUNDS, nodata=None, dtype="uint8"):
    from rasterio.transform import from_bounds
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_bounds(*bounds, arr.shape[1], arr.shape[0]), nodata=nodata) as d:
        d.write(arr.astype(dtype)[np.newaxis])
    return str(path)


def _pipe(monkeypatch, outputs, **kw):
    """A pipeline whose sources are replaced by canned results."""
    p = AutoLabelPipeline(sources=list(outputs), **kw)
    monkeypatch.setattr(p, "_run_source", lambda source, bbox, out, cats: outputs[source])
    return p


def _ok(path):
    return {"success": True, "output_path": path}


def test_majority_vote_of_three_building_maps(monkeypatch, tmp_path):
    a = np.zeros((10, 10), "uint8"); a[:5] = 1           # rows 0-4
    b = np.zeros((10, 10), "uint8"); b[:3] = 1           # rows 0-2
    c = np.zeros((10, 10), "uint8"); c[2:5] = 1          # rows 2-4
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "a.tif", a)),
                            "microsoft_buildings": _ok(_raster(tmp_path / "b.tif", b)),
                            "google_buildings": _ok(_raster(tmp_path / "c.tif", c))})
    r = p.run(BOUNDS, str(tmp_path), assess_quality=False)
    assert r["fused_path"] and r["fusion_skipped_reason"] is None
    with rasterio.open(r["fused_path"]) as f:
        out = f.read(1)
    expected = np.zeros((10, 10), "uint8")
    expected[:5] = 1                      # rows 0-1: a+b; row 2: a+b+c; rows 3-4: a+c  -> all majority
    assert (out == expected).all()


def test_tie_goes_to_the_earlier_source(monkeypatch, tmp_path):
    a = np.ones((4, 4), "uint8")
    b = np.zeros((4, 4), "uint8")
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "a.tif", a)),
                            "microsoft_buildings": _ok(_raster(tmp_path / "b.tif", b))})
    with rasterio.open(p.run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        assert (f.read(1) == 1).all()
    p = _pipe(monkeypatch, {"microsoft_buildings": _ok(str(tmp_path / "b.tif")), "osm": _ok(str(tmp_path / "a.tif"))})
    with rasterio.open(p.run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        assert (f.read(1) == 0).all()


def test_different_resolutions_are_aligned_by_georeference(monkeypatch, tmp_path):
    fine = np.zeros((100, 100), "uint8"); fine[:50, :50] = 1            # top-left quarter
    coarse = np.zeros((25, 25), "uint8"); coarse[:12, :12] = 1          # same area, coarser
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "f.tif", fine)),
                            "microsoft_buildings": _ok(_raster(tmp_path / "c.tif", coarse))}, fusion="union")
    r = p.run(BOUNDS, str(tmp_path), assess_quality=False)
    with rasterio.open(r["fused_path"]) as f:
        out = f.read(1)
        assert out.shape == (100, 100) and f.nodata == 255
    assert out[:50, :50].all() and out[60:, :].sum() == 0 and out[:, 60:].sum() == 0
    assert 2500 <= int(out.sum()) <= 2900       # union of the fine quarter and the coarser (12/25 -> 48 px) one


def test_reference_raster_decides_the_output_grid(monkeypatch, tmp_path):
    a = np.ones((10, 10), "uint8"); b = np.ones((10, 10), "uint8")
    ref = _raster(tmp_path / "ref.tif", np.zeros((30, 40), "uint8"))
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "a.tif", a)),
                            "microsoft_buildings": _ok(_raster(tmp_path / "b.tif", b))}, reference_raster=ref)
    with rasterio.open(p.run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        assert f.shape == (30, 40)


def test_nodata_pixels_abstain(monkeypatch, tmp_path):
    a = np.ones((6, 6), "uint8")
    b = np.zeros((6, 6), "uint8"); b[:, 3:] = 255
    c = np.zeros((6, 6), "uint8")
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "a.tif", a)),
                            "microsoft_buildings": _ok(_raster(tmp_path / "b.tif", b, nodata=255)),
                            "google_buildings": _ok(_raster(tmp_path / "c.tif", c))})
    with rasterio.open(p.run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        out = f.read(1)
    assert (out[:, :3] == 0).all()        # a=1, b=0, c=0 -> 0
    assert (out[:, 3:] == 1).all()        # b abstains: a=1, c=0 tie -> earlier source (a)


def test_incompatible_alphabets_are_not_fused(monkeypatch, tmp_path):
    osm = np.zeros((10, 10), "uint8"); osm[:3] = 1; osm[3:6] = 2
    esa = np.full((10, 10), 50, "uint8")
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "o.tif", osm)),
                            "esa_worldcover": _ok(_raster(tmp_path / "e.tif", esa))})
    r = p.run(BOUNDS, str(tmp_path))
    assert r["success"] and r["fused_path"] is None
    assert "different label alphabets" in r["fusion_skipped_reason"]
    assert set(r["quality_by_source"]) == {"osm", "esa_worldcover"}
    assert r["label_paths"] and "quality" not in r


def test_sam_instances_are_never_fused(monkeypatch, tmp_path):
    a = np.zeros((6, 6), "uint8"); a[:2] = 1
    p = _pipe(monkeypatch, {"osm": _ok(_raster(tmp_path / "a.tif", a)),
                            "sam_auto": _ok(_raster(tmp_path / "s.tif", a))})
    r = p.run(BOUNDS, str(tmp_path), assess_quality=False)
    assert r["fused_path"] is None and "arbitrary ids" in r["fusion_skipped_reason"]


def test_single_source_is_returned_as_is_and_assessed(monkeypatch, tmp_path):
    a = np.zeros((20, 20), "uint8"); a[5:15, 5:15] = 1
    path = _raster(tmp_path / "a.tif", a)
    p = _pipe(monkeypatch, {"osm": _ok(path), "microsoft_buildings": {"success": False, "error": "no coverage"}})
    r = p.run(BOUNDS, str(tmp_path))
    assert r["fused_path"] == path and r["sources_failed"] == ["microsoft_buildings"]
    assert "quality_score" in r["quality"]


def test_async_export_does_not_count_as_a_label(monkeypatch, tmp_path):
    p = _pipe(monkeypatch, {"dynamic_world": {"success": True, "output_path": None, "submitted": True}})
    r = p.run(BOUNDS, str(tmp_path), assess_quality=False)
    assert not r["success"] and r["sources_failed"] == ["dynamic_world"]
    assert r["partial_results"]["dynamic_world"]["success"] is False


def test_constructor_validation():
    with pytest.raises(ValueError, match="fusion"):
        AutoLabelPipeline(fusion="averge")
    with pytest.raises(ValueError, match="Unknown source"):
        AutoLabelPipeline(sources=["osm", "bing"])


def test_priority_and_intersection(monkeypatch, tmp_path):
    a = np.zeros((4, 4), "uint8"); a[:2] = 1
    b = np.zeros((4, 4), "uint8"); b[1:3] = 1
    outs = {"osm": _ok(_raster(tmp_path / "a.tif", a)), "microsoft_buildings": _ok(_raster(tmp_path / "b.tif", b))}
    with rasterio.open(_pipe(monkeypatch, outs, fusion="intersection").run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        assert f.read(1).sum() == 4                      # only row 1
    with rasterio.open(_pipe(monkeypatch, outs, fusion="priority").run(BOUNDS, str(tmp_path), assess_quality=False)["fused_path"]) as f:
        assert f.read(1).sum() == 12                     # rows 0-2
