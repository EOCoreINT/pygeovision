"""`ai detect` must refuse a raster output path for box detectors and surface detector errors."""
import importlib

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from click.testing import CliRunner  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.cli.main import cli  # noqa: E402


@pytest.fixture()
def scene(tmp_path):
    p = tmp_path / "in.tif"
    with rasterio.open(p, "w", driver="GTiff", height=32, width=32, count=3, dtype="uint8",
                       crs="EPSG:32630", transform=from_origin(0, 320, 10, 10)) as d:
        d.write(np.zeros((3, 32, 32), "uint8"))
    return p


def _run(scene, out, target="generic"):
    return CliRunner().invoke(cli, ["ai", "detect", target, "--input", str(scene), "--output", str(out)])


def test_tif_output_is_refused_for_box_detectors(scene, tmp_path):
    r = _run(scene, tmp_path / "d.tif")
    assert r.exit_code != 0
    assert "GeoJSON" in r.output


def test_detector_error_is_reported_not_swallowed(scene, tmp_path, monkeypatch):
    mod = importlib.import_module("pygeovision")
    monkeypatch.setattr(mod._DetectionClientProxy, "generic", lambda self, *a, **k: {"error": "pip install ultralytics"})
    r = _run(scene, tmp_path / "d.geojson")
    assert r.exit_code != 0
    assert "ultralytics" in r.output
    assert "Detection complete" not in r.output


def test_success_reports_the_count(scene, tmp_path, monkeypatch):
    mod = importlib.import_module("pygeovision")
    monkeypatch.setattr(mod._DetectionClientProxy, "ships",
                        lambda self, *a, **k: {"n_detections": 3, "detections": []})
    r = _run(scene, tmp_path / "s.geojson", target="ships")
    assert r.exit_code == 0, r.output
    assert "3 object(s) found" in r.output
