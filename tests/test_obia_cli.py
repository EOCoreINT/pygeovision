"""End-to-end test of the `pygeovision obia` command group."""
import textwrap

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("rasterio")
pytest.importorskip("geopandas")
from click.testing import CliRunner  # noqa: E402

from pygeovision import obia  # noqa: E402
from pygeovision.cli.main import cli  # noqa: E402
from tests.obia_testdata import CLASSES, block_scene  # noqa: E402


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """scene.tif, reference.gpkg and a WorldCover-style class map written from the hand-built block scene."""
    d = tmp_path_factory.mktemp("obia_cli")
    sc = block_scene(64)
    r = sc.raster
    obia.save_raster(str(d / "scene.tif"), r.image, r.transform, r.crs, r.band_names)
    sc.reference.to_file(d / "reference.gpkg", driver="GPKG")
    codes = {"water": 80, "cropland": 40, "bare_field": 60, "forest": 10, "building": 50}   # WorldCover codes
    wc = np.zeros(sc.ref_labels.shape, np.uint8)
    for i, c in enumerate(CLASSES):
        wc[sc.class_raster == i + 1] = codes[c]
    obia.save_raster(str(d / "wc.tif"), wc, r.transform, r.crs, dtype="uint8")
    return d


def _run(*args):
    r = CliRunner().invoke(cli, list(args))
    assert r.exit_code == 0, r.output
    return r.output


def test_documented_workflow(workdir):
    d = str(workdir)
    p = lambda n: f"{d}/{n}"
    assert "objects (multires)" in _run("obia", "segment", p("scene.tif"), "--scale", "20",
                                        "--output-labels", p("labels.tif"), "--output-vector", p("objects.gpkg"))
    out = _run("obia", "scale", p("scene.tif"), "--scales", "10,20,40", "--reference", p("reference.gpkg"))
    assert "best scale by D" in out
    _run("obia", "features", p("scene.tif"), p("labels.tif"), "--output", p("f.csv"), "--texture", "nir")
    import pandas as pd
    F = pd.read_csv(p("f.csv"))
    assert "glcm_contrast_nir" in F.columns and "mean_ndvi" in F.columns
    out = _run("obia", "classify", p("scene.tif"), p("labels.tif"), "--samples", p("reference.gpkg"),
               "--output", p("c.gpkg"), "--output-raster", p("c.tif"))
    assert "NOT used for training" in out

    with open(p("rules.yaml"), "w") as fh:
        fh.write(textwrap.dedent("""
            rules:
              - {class: water,      when: "mean_ndwi > 0.4"}
              - {class: bare_field, when: "mean_ndvi <= 0.5"}
              - {class: cropland,   when: "mean_ndvi > 0.5"}
        """))
    _run("obia", "classify", p("scene.tif"), p("labels.tif"), "--rules", p("rules.yaml"), "--output", p("r.gpkg"))
    assert "mean D" in _run("obia", "assess-segmentation", p("labels.tif"), p("reference.gpkg"))
    assert "overall accuracy" in _run("obia", "assess-classification", p("labels.tif"), p("r.gpkg"),
                                      p("reference.gpkg"))


def test_classify_requires_exactly_one_source(workdir):
    d = str(workdir)
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--output", f"{d}/x.gpkg"])
    assert r.exit_code == 2 and "exactly one of --samples, --weak-labels or --rules" in r.output


def test_bad_rule_is_a_clean_error_not_a_traceback(workdir):
    d = str(workdir)
    if not __import__("os").path.exists(f"{d}/labels.tif"):
        _run("obia", "segment", f"{d}/scene.tif", "--scale", "20", "--output-labels", f"{d}/labels.tif")
    with open(f"{d}/bad.yaml", "w") as fh:
        fh.write('rules:\n  - {class: water, when: "mean_ndwii > 0.3"}\n')
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--rules",
                                 f"{d}/bad.yaml", "--output", f"{d}/y.gpkg"])
    assert r.exit_code == 1 and "cannot evaluate rule" in r.output and "available columns" in r.output
    assert "Traceback" not in r.output
    with open(f"{d}/worse.yaml", "w") as fh:
        fh.write("nothing: here\n")
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--rules",
                                 f"{d}/worse.yaml", "--output", f"{d}/y.gpkg"])
    assert r.exit_code == 1 and "expected 'rules:" in r.output


def test_classify_from_a_weak_label_raster(workdir):
    d = str(workdir)
    if not __import__("os").path.exists(f"{d}/labels.tif"):
        _run("obia", "segment", f"{d}/scene.tif", "--scale", "20", "--output-labels", f"{d}/labels.tif")
    out = _run("obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--weak-labels", f"{d}/wc.tif",
               "--class-map", "80=water,40=cropland,60=bare_field,10=forest,50=building",
               "--min-purity", "0.9", "--output", f"{d}/w.gpkg", "--output-raster", f"{d}/w.tif")
    assert "trained on 5 sample objects (5 classes)" in out and "independent of the weak-label map" in out
    import geopandas as gpd
    got = gpd.read_file(f"{d}/w.gpkg").set_index("obj_id")["class"]
    assert sorted(got.unique()) == sorted(CLASSES)
    # the class map is mandatory, and malformed maps are usage errors
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--weak-labels",
                                 f"{d}/wc.tif", "--output", f"{d}/z.gpkg"])
    assert r.exit_code == 2 and "--class-map" in r.output
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--weak-labels",
                                 f"{d}/wc.tif", "--class-map", "ten=forest", "--output", f"{d}/z.gpkg"])
    assert r.exit_code == 2 and "code=class" in r.output
    # a purity nobody reaches is a clean error, not an empty file
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--weak-labels",
                                 f"{d}/wc.tif", "--class-map", "99=forest", "--output", f"{d}/z.gpkg"])
    assert r.exit_code == 1 and "purity" in r.output and "Traceback" not in r.output


def test_samples_in_the_wrong_crs_are_reprojected_and_a_wrong_extent_is_a_clean_error(workdir):
    import geopandas as gpd
    d = str(workdir)
    ref = gpd.read_file(f"{d}/reference.gpkg")
    ref.to_crs(4326).to_file(f"{d}/reference_ll.gpkg", driver="GPKG")
    _run("obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--samples", f"{d}/reference_ll.gpkg",
         "--output", f"{d}/ll.gpkg")
    far = ref.copy()
    far["geometry"] = far.geometry.translate(5_000_000, 0)
    far.to_file(f"{d}/far.gpkg", driver="GPKG")
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--samples",
                                 f"{d}/far.gpkg", "--output", f"{d}/far_out.gpkg"])
    assert r.exit_code == 1 and "outside the image" in r.output and "Traceback" not in r.output


def test_fetch_writes_the_scene_and_worldcover_and_fails_cleanly(workdir, monkeypatch):
    d = str(workdir)
    sc = block_scene(64)
    calls = {}

    def fake_scene(client, bbox, date, out_dir, bands, cloud_cover_max, **kw):
        calls.update(bbox=bbox, date=date, bands=list(bands), cloud=cloud_cover_max, kw=kw)
        img = sc.raster.image.copy()
        img[:, :8, :8] = np.nan
        return obia.RasterData(img, list(bands)[:4], sc.raster.transform, sc.raster.crs)

    def fake_wc(client, bbox, out_path, like=None, **kw):
        obia.save_raster(str(out_path), np.full(like.shape, 10, np.uint8), like.transform, like.crs, dtype="uint8")
        return np.full(like.shape, 10, np.uint8), out_path
    monkeypatch.setattr(obia, "fetch_scene", fake_scene)
    monkeypatch.setattr(obia, "fetch_worldcover", fake_wc)
    out = _run("obia", "fetch", "--bbox", "-1.70", "6.18", "-1.655", "6.225", "--date", "2024-01", "-o", f"{d}/f.tif",
               "--bands", "blue,green,red,nir", "--cloud-max", "15", "--providers", "planetary_computer",
               "--worldcover", f"{d}/fwc.tif")
    assert calls["bbox"] == (-1.7, 6.18, -1.655, 6.225) and calls["date"] == "2024-01"
    assert calls["bands"] == ["blue", "green", "red", "nir"] and calls["cloud"] == 15.0
    assert calls["kw"] == {"providers": ["planetary_computer"], "verbose": True, "heartbeat": 10.0}
    assert "no data (cloud/shadow): 1.6%" in out and "WorldCover on the image grid" in out      # 64 of 4096 px
    # --quiet turns the progress off and heartbeat is passed on
    _run("obia", "fetch", "--bbox", "0", "0", "1", "1", "--date", "2024-01", "-o", f"{d}/q.tif", "--quiet", "--heartbeat", "3")
    assert calls["kw"] == {"verbose": False, "heartbeat": 3.0}
    rd = obia.load_raster(f"{d}/f.tif")
    assert rd.band_names == ["blue", "green", "red", "nir"] and np.isnan(rd.image[:, :8, :8]).all()
    # the saved scene goes straight into segmentation: the NaN corner is label 0
    _run("obia", "segment", f"{d}/f.tif", "--scale", "20", "--output-labels", f"{d}/fl.tif")
    import rasterio
    with rasterio.open(f"{d}/fl.tif") as src:
        lab = src.read(1)
    assert (lab[:8, :8] == 0).all() and (lab[8:, 8:] > 0).all()

    def boom(*a, **k):
        raise RuntimeError("no usable Sentinel-2 scene for bbox=...")
    monkeypatch.setattr(obia, "fetch_scene", boom)
    r = CliRunner().invoke(cli, ["obia", "fetch", "--bbox", "0", "0", "1", "1", "--date", "2024-01", "-o", f"{d}/x.tif"])
    assert r.exit_code == 1 and "no usable Sentinel-2 scene" in r.output and "Traceback" not in r.output
    assert not __import__("os").path.exists(f"{d}/x.tif")


def test_there_is_no_demo_command():
    r = CliRunner().invoke(cli, ["obia", "demo", "x"])
    assert r.exit_code != 0 and "No such command" in r.output
