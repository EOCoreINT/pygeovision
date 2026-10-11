"""Regression tests for LabelQualityAssessor.

Real defects fixed:
  1. Pixel area was ``|a*e|`` of the geotransform. For a lon/lat raster (the default OSM output!) that is
     ~1e-10 "m2", so ``clean()`` computed a minimum component of billions of pixels and **erased every
     label**; ``assess()`` called every component a sliver.
  2. Classes beyond ``num_classes`` (OSM uses 1..10) were silently ignored.
  3. ``connectivity`` used only the last class, crashed for ``num_classes=1`` and scored "few components"
     as good, so a map of 1,000 buildings scored 0.01.
  4. ``clean()`` looped over every component with a full-raster comparison (O(components x pixels)) and
     let morphological closing overwrite neighbouring classes.
  5. ``before/after_label_pixels`` counted the no-data value 255 as foreground.
  6. Unknown ``checks`` names were silently ignored.
"""
import time

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pytest.importorskip("scipy")

from pygeovision.labeling.quality import LabelQualityAssessor  # noqa: E402


def _write(path, arr, crs="EPSG:4326", res=0.0001, nodata=None):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="uint8", crs=crs, transform=from_origin(10.0, 50.0, res, res), nodata=nodata) as d:
        d.write(arr.astype("uint8")[np.newaxis])
    return str(path)


def test_clean_keeps_real_objects_on_a_lonlat_raster(tmp_path):
    a = np.zeros((60, 60), "uint8")
    a[10:16, 10:16] = 1          # 36 px object
    a[40, 40] = 1                # 1 px speck
    a[30:32, 30:32] = 1          # 4 px speck
    src = _write(tmp_path / "l.tif", a)           # 0.0001 deg ~ 7 m x 11 m ~ 80 m2 per pixel
    res = LabelQualityAssessor().clean(src, tmp_path / "c.tif", min_area_m2=250.0, fill_holes=False, smooth=False)
    assert res["min_component_px"] == 4           # ceil(250 / ~79)
    assert res["n_components_removed"] == 1       # only the 1 px speck is < 4 px
    assert res["before_label_pixels"] == 41 and res["after_label_pixels"] == 40
    with rasterio.open(tmp_path / "c.tif") as o:
        out = o.read(1)
    assert out[10:16, 10:16].all() and out[40, 40] == 0 and out[30:32, 30:32].all()


def test_pixel_area_in_metres_for_each_crs(tmp_path):
    from pygeovision.labeling.quality import pixel_area_m2
    g = _write(tmp_path / "g.tif", np.zeros((10, 10)), res=0.0001)
    p = _write(tmp_path / "p.tif", np.zeros((10, 10)), crs="EPSG:32632", res=10.0)
    with rasterio.open(g) as s:
        assert 60 < pixel_area_m2(s) < 90
    with rasterio.open(p) as s:
        assert pixel_area_m2(s) == pytest.approx(100.0)


def test_multiclass_values_are_counted_and_unexpected_values_reported(tmp_path):
    a = np.zeros((40, 40), "uint8")
    a[0:10, 0:10] = 1
    a[20:30, 20:30] = 2
    a[0:5, 30:40] = 3
    src = _write(tmp_path / "m.tif", a, crs="EPSG:32632", res=10.0)
    r = LabelQualityAssessor().assess(src)
    assert r["classes"] == [0, 1, 2, 3] and r["unexpected_values"] == []
    assert set(r["checks"]["class_balance"]["class_counts"]) == {0, 1, 2, 3}
    r2 = LabelQualityAssessor(num_classes=2).assess(src)
    assert r2["unexpected_values"] == [2, 3]
    assert any("outside 0..1" in x for x in r2["recommendations"])


def test_connectivity_does_not_punish_many_large_objects(tmp_path):
    a = np.zeros((400, 400), "uint8")
    for i in range(0, 400, 20):
        for j in range(0, 400, 20):
            a[i + 3:i + 10, j + 3:j + 10] = 1          # 400 objects of 49 px
    src = _write(tmp_path / "b.tif", a, crs="EPSG:32632", res=1.0)
    r = LabelQualityAssessor().assess(src)
    assert r["checks"]["connectivity"]["n_components"] == 400
    assert r["checks"]["connectivity"]["score"] == 1.0

    dust = np.zeros((400, 400), "uint8")
    dust[::3, ::3] = 1
    r = LabelQualityAssessor().assess(_write(tmp_path / "d.tif", dust, crs="EPSG:32632", res=1.0))
    assert r["checks"]["connectivity"]["score"] < 0.1
    assert r["checks"]["connectivity"]["status"] == "warning"
    assert r["checks"]["slivers"]["status"] == "warning"


def test_single_class_and_empty_rasters_do_not_crash(tmp_path):
    r = LabelQualityAssessor(num_classes=1).assess(_write(tmp_path / "e.tif", np.zeros((20, 20)), crs="EPSG:32632", res=1))
    assert r["quality_grade"] == "F"
    assert any("Only one class" in x for x in r["recommendations"])


def test_nodata_is_not_foreground_and_is_never_edited(tmp_path):
    a = np.zeros((30, 30), "uint8")
    a[:, 20:] = 255
    a[5:12, 5:12] = 1
    src = _write(tmp_path / "n.tif", a, crs="EPSG:32632", res=1.0, nodata=255)
    res = LabelQualityAssessor().clean(src, tmp_path / "c.tif")
    assert res["before_label_pixels"] == 49 and res["after_label_pixels"] == 49
    with rasterio.open(tmp_path / "c.tif") as o:
        out = o.read(1)
        assert o.nodata == 255
    assert (out[:, 20:] == 255).all()
    r = LabelQualityAssessor().assess(src)
    assert r["checks"]["coverage"]["valid_fraction"] == pytest.approx(20 / 30, abs=0.01)


def test_smoothing_never_overwrites_another_class(tmp_path):
    a = np.zeros((30, 30), "uint8")
    a[5:20, 5:12] = 1
    a[5:20, 13:20] = 2           # one background column (12) between them
    src = _write(tmp_path / "t.tif", a, crs="EPSG:32632", res=1.0)
    LabelQualityAssessor().clean(src, tmp_path / "c.tif", smooth=True, fill_holes=False, remove_slivers=False)
    with rasterio.open(tmp_path / "c.tif") as o:
        out = o.read(1)
    assert (out[5:20, 5:12] == 1).all() and (out[5:20, 13:20] == 2).all()


def test_clean_is_fast_with_many_components(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((1500, 1500)) > 0.9).astype("uint8")       # ~100k specks
    src = _write(tmp_path / "x.tif", a, crs="EPSG:32632", res=1.0)
    t = time.time()
    LabelQualityAssessor().clean(src, tmp_path / "c.tif", fill_holes=False, smooth=False)
    assert time.time() - t < 20


def test_unknown_check_raises(tmp_path):
    src = _write(tmp_path / "u.tif", np.zeros((5, 5)), crs="EPSG:32632", res=1.0)
    with pytest.raises(ValueError, match="Unknown check"):
        LabelQualityAssessor().assess(src, checks=["sliver"])


def test_boundaries_check_scores_compact_shapes_above_ragged_ones(tmp_path):
    pytest.importorskip("cv2")
    sq = np.zeros((60, 60), "uint8")
    sq[10:40, 10:40] = 1
    rag = np.zeros((60, 60), "uint8")
    rag[10:40, 10:40] = 1
    rag[10:40:2, 40:55] = 1          # comb teeth
    s = LabelQualityAssessor().assess(_write(tmp_path / "s.tif", sq, crs="EPSG:32632", res=1.0), checks=["boundaries"])
    r = LabelQualityAssessor().assess(_write(tmp_path / "r.tif", rag, crs="EPSG:32632", res=1.0), checks=["boundaries"])
    assert s["checks"]["boundaries"]["isoperimetric_quotient"] > r["checks"]["boundaries"]["isoperimetric_quotient"]
    assert s["quality_score"] == s["checks"]["boundaries"]["score"]


def test_html_report_escapes_user_text():
    html = LabelQualityAssessor().report_html({"label_path": "<script>x</script>.tif", "quality_grade": "B",
                                               "quality_score": 0.8, "checks": {}, "recommendations": ["a<b"]})
    assert "<script>" not in html and "a&lt;b" in html


def test_cli_quality_options(tmp_path):
    from click.testing import CliRunner

    from pygeovision.cli.main import cli
    a = np.zeros((40, 40), "uint8")
    a[5:20, 5:20] = 1
    a[25:35, 25:35] = 2
    src = _write(tmp_path / "q.png.tif", a, crs="EPSG:32632", res=1.0)
    r = CliRunner().invoke(cli, ["label", "quality", src, "--check", "coverage", "--html"])
    assert r.exit_code == 0, r.output
    assert "coverage" in r.output and "class_balance" not in r.output
    assert (tmp_path / "q.png_quality.html").exists()
    r = CliRunner().invoke(cli, ["label", "quality", str(tmp_path / "missing.tif")])
    assert r.exit_code != 0 and "cannot read" in r.output
