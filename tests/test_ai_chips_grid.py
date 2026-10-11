"""`ai chips` must refuse an image/label pair that is not on the same grid."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from click.testing import CliRunner  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.cli.main import cli  # noqa: E402


def _write(path, arr, transform, crs="EPSG:32630"):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                       count=arr.shape[0], dtype=str(arr.dtype), crs=crs, transform=transform) as d:
        d.write(arr)


def test_matching_grid_exports_chips(tmp_path):
    t = from_origin(500000, 700000, 10, 10)
    _write(tmp_path / "i.tif", np.random.rand(3, 64, 64).astype("float32"), t)
    _write(tmp_path / "l.tif", np.zeros((1, 64, 64), "uint8"), t)
    r = CliRunner().invoke(cli, ["ai", "chips", "--image", str(tmp_path / "i.tif"),
                                  "--label", str(tmp_path / "l.tif"), "-o", str(tmp_path / "o"),
                                  "--chip-size", "32"])
    assert r.exit_code == 0, r.output
    assert "4 chips exported" in r.output


@pytest.mark.parametrize("label_transform,label_size", [
    (from_origin(500100, 700000, 10, 10), 64),   # shifted origin
    (from_origin(500000, 700000, 10, 10), 48),   # different size
])
def test_mismatched_grid_is_refused(tmp_path, label_transform, label_size):
    _write(tmp_path / "i.tif", np.random.rand(3, 64, 64).astype("float32"), from_origin(500000, 700000, 10, 10))
    _write(tmp_path / "l.tif", np.zeros((1, label_size, label_size), "uint8"), label_transform)
    r = CliRunner().invoke(cli, ["ai", "chips", "--image", str(tmp_path / "i.tif"),
                                  "--label", str(tmp_path / "l.tif"), "-o", str(tmp_path / "o"),
                                  "--chip-size", "32"])
    assert r.exit_code != 0
    assert "not on the same grid" in r.output
    assert not list((tmp_path / "o" / "images").glob("*.tif"))


def test_prefix_keeps_two_scenes_apart(tmp_path):
    t = from_origin(500000, 700000, 10, 10)
    _write(tmp_path / "i.tif", np.random.rand(3, 64, 64).astype("float32"), t)
    _write(tmp_path / "l.tif", np.zeros((1, 64, 64), "uint8"), t)
    for pre in ("a", "b"):
        r = CliRunner().invoke(cli, ["ai", "chips", "--image", str(tmp_path / "i.tif"),
                                     "--label", str(tmp_path / "l.tif"), "-o", str(tmp_path / "o"),
                                     "--chip-size", "32", "--prefix", pre])
        assert r.exit_code == 0, r.output
    names = sorted(p.name for p in (tmp_path / "o" / "images").glob("*.tif"))
    assert len(names) == 8 and names[0].startswith("a_") and names[-1].startswith("b_")
