"""Clipping outside the raster must say so, in the user's terms."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


@pytest.fixture
def scene(tmp_path):
    p = tmp_path / "s.tif"
    with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=2, dtype="float32", crs="EPSG:32633",
                       transform=from_origin(500000, 5300000, 10, 10), tiled=True, blockxsize=16, blockysize=16) as d:
        d.write(np.random.default_rng(0).uniform(0, 1, (2, 64, 64)).astype("float32"))
    return str(p)


def test_bbox_elsewhere_names_the_raster_extent(scene, tmp_path):
    from pygeovision.preprocess import Preprocessor
    with pytest.raises(ValueError, match=r"does not overlap the raster.*longitude 15\.0000 to"):
        Preprocessor().clip_to_bbox(scene, (12.4, 47.8, 12.45, 47.85), str(tmp_path / "o.tif"))


def test_inverted_bbox_rejected(scene, tmp_path):
    from pygeovision.preprocess import Preprocessor
    with pytest.raises(ValueError, match="min < max"):
        Preprocessor().clip_to_bbox(scene, (15.01, 47.84, 15.0, 47.83), str(tmp_path / "o.tif"))


def test_overlapping_bbox_still_clips(scene, tmp_path):
    from pygeovision.preprocess import Preprocessor
    out = Preprocessor().clip_to_bbox(scene, (15.0, 47.849, 15.003, 47.852), str(tmp_path / "o.tif"))
    with rasterio.open(out) as s:
        assert 0 < s.width < 64 and 0 < s.height < 64


def test_polygon_elsewhere_explained(scene, tmp_path):
    from pygeovision.preprocess import Preprocessor
    poly = {"type": "Polygon", "coordinates": [[[12.4, 47.8], [12.45, 47.8], [12.45, 47.85], [12.4, 47.8]]]}
    with pytest.raises(ValueError, match="polygon does not overlap|The polygon does not overlap"):
        Preprocessor().clip_to_polygon(scene, poly, str(tmp_path / "o.tif"))


def test_cli_messages(scene, tmp_path):
    from click.testing import CliRunner

    from pygeovision.cli.main import cli
    r = CliRunner().invoke(cli, ["preprocess", "clip", scene, "--bbox", "12.4,47.8,12.45,47.85"])
    assert r.exit_code != 0 and "does not overlap" in r.output and "Traceback" not in r.output
    r = CliRunner().invoke(cli, ["preprocess", "clip", scene, "--bbox", "1,2,3"])
    assert r.exit_code != 0 and "four numbers" in r.output
