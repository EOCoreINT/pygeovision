"""End-to-end test of the `pygeovision obia` command group."""
import textwrap

import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("rasterio")
pytest.importorskip("geopandas")
from click.testing import CliRunner  # noqa: E402

from pygeovision.cli.main import cli  # noqa: E402


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("obia_cli")


def _run(*args):
    r = CliRunner().invoke(cli, list(args))
    assert r.exit_code == 0, r.output
    return r.output


def test_documented_workflow(workdir):
    d = str(workdir)
    p = lambda n: f"{d}/{n}"
    assert "Demo scene" in _run("obia", "demo", d, "--size", "112")
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
              - {class: water,      when: "mean_ndwi > 0.3"}
              - {class: bare_field, when: "mean_ndvi <= 0.55"}
              - {class: cropland,   when: "mean_ndvi > 0.55"}
        """))
    _run("obia", "classify", p("scene.tif"), p("labels.tif"), "--rules", p("rules.yaml"), "--output", p("r.gpkg"))
    assert "mean D" in _run("obia", "assess-segmentation", p("labels.tif"), p("reference.gpkg"))
    assert "overall accuracy" in _run("obia", "assess-classification", p("labels.tif"), p("r.gpkg"),
                                      p("reference.gpkg"))


def test_classify_requires_exactly_one_source(workdir):
    d = str(workdir)
    r = CliRunner().invoke(cli, ["obia", "classify", f"{d}/scene.tif", f"{d}/labels.tif", "--output", f"{d}/x.gpkg"])
    assert r.exit_code == 2 and "exactly one" in r.output


def test_bad_rule_is_a_clean_error_not_a_traceback(workdir):
    d = str(workdir)
    if not __import__("os").path.exists(f"{d}/labels.tif"):
        _run("obia", "demo", d, "--size", "112")
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
