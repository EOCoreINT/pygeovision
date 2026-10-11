"""TiledInference.infer must treat NaN / nodata pixels as 'no data' instead of predicting them."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
torch = pytest.importorskip("torch")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.inference.tiled import TiledInference  # noqa: E402


class _Threshold(torch.nn.Module):
    """class 1 where band 0 > 0.5."""

    def forward(self, x):
        v = x[:, :1]
        return torch.cat([0.5 - v, v - 0.5], 1) * 10


def _write(path, arr, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:32630", transform=from_origin(0, 640, 10, 10), nodata=nodata) as d:
        d.write(arr)


def _engine():
    return TiledInference(_Threshold(), chip_size=32, overlap=0, num_classes=2, device="cpu")


def test_nan_pixels_do_not_blank_the_map_and_are_marked_nodata(tmp_path):
    arr = np.full((2, 64, 64), 0.2, "float32")
    arr[:, :, 32:] = 0.8
    arr[:, :8, :8] = np.nan
    _write(tmp_path / "in.tif", arr)
    res = _engine().infer(tmp_path / "in.tif", tmp_path / "out.tif", normalise=False)
    with rasterio.open(tmp_path / "out.tif") as r:
        lab, nodata = r.read(1), r.nodata
    assert nodata == 255
    assert (lab[:8, :8] == 255).all()
    assert (lab[16:, 40:] == 1).all() and (lab[16:, :24] == 0).all()
    assert res["valid_fraction"] == pytest.approx(1 - 64 / 4096, abs=1e-4)


def test_stretch_ignores_nan_and_nodata_pixels(tmp_path):
    arr = np.linspace(0.2, 0.8, 64 * 64, dtype="float32").reshape(1, 64, 64).repeat(2, 0)
    arr[:, :4, :] = np.nan
    _write(tmp_path / "in.tif", arr)
    _engine().infer(tmp_path / "in.tif", tmp_path / "out.tif", normalise=True)
    with rasterio.open(tmp_path / "out.tif") as r:
        lab = r.read(1)
    assert (lab[:4] == 255).all()
    assert set(np.unique(lab[4:])) == {0, 1}          # the stretch worked: both classes appear


def test_declared_nodata_value_is_honoured(tmp_path):
    arr = np.full((2, 64, 64), 0.8, "float32")
    arr[:, 32:, :] = 0.0
    _write(tmp_path / "in.tif", arr, nodata=0.0)
    _engine().infer(tmp_path / "in.tif", tmp_path / "out.tif", normalise=False)
    with rasterio.open(tmp_path / "out.tif") as r:
        lab = r.read(1)
    assert (lab[32:] == 255).all() and (lab[:32] == 1).all()


def test_explicit_nodata_value_argument(tmp_path):
    arr = np.full((2, 64, 64), 0.8, "float32")
    arr[:, :, :16] = -9999.0
    _write(tmp_path / "in.tif", arr)
    _engine().infer(tmp_path / "in.tif", tmp_path / "out.tif", normalise=False, nodata_value=-9999.0)
    with rasterio.open(tmp_path / "out.tif") as r:
        lab = r.read(1)
    assert (lab[:, :16] == 255).all() and (lab[:, 16:] == 1).all()


def test_complete_scene_keeps_no_nodata_flag(tmp_path):
    arr = np.full((2, 64, 64), 0.8, "float32")
    _write(tmp_path / "in.tif", arr)
    res = _engine().infer(tmp_path / "in.tif", tmp_path / "out.tif", normalise=False)
    with rasterio.open(tmp_path / "out.tif") as r:
        assert r.nodata is None
    assert res["valid_fraction"] == 1.0
