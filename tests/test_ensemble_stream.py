"""EnsembleInference: fusion validation, no temporary rasters left behind, nodata kept; Streaming delegates."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
torch = pytest.importorskip("torch")
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.inference.stream import EnsembleInference, StreamingInference  # noqa: E402


class _Const(torch.nn.Module):
    def __init__(self, cls, strength=4.0):
        super().__init__()
        self.cls, self.strength = cls, strength

    def forward(self, x):
        out = torch.zeros(x.shape[0], 2, *x.shape[2:])
        out[:, self.cls] = self.strength
        return out


def _scene(path, nan_block=False):
    arr = np.random.rand(2, 64, 64).astype("float32")
    if nan_block:
        arr[:, :8, :8] = np.nan
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=2, dtype="float32",
                       crs="EPSG:32630", transform=from_origin(0, 640, 10, 10)) as d:
        d.write(arr)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="fusion"):
        EnsembleInference([_Const(0)], fusion="median")
    with pytest.raises(ValueError, match="weights"):
        EnsembleInference([_Const(0), _Const(1)], weights=[1.0])
    with pytest.raises(ValueError, match="at least one"):
        EnsembleInference([])


def test_weights_decide_the_winner_and_no_temp_files_remain(tmp_path):
    _scene(tmp_path / "in.tif")
    ens = EnsembleInference([_Const(0), _Const(1)], weights=[0.2, 0.8], device="cpu", normalise=False)
    res = ens.infer(tmp_path / "in.tif", tmp_path / "out.tif")
    assert res["success"]
    with rasterio.open(tmp_path / "out.tif") as r:
        assert (r.read(1) == 1).all()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.tif", "out.tif"]


def test_nodata_pixels_stay_nodata(tmp_path):
    _scene(tmp_path / "in.tif", nan_block=True)
    EnsembleInference([_Const(1)], device="cpu", normalise=False).infer(tmp_path / "in.tif", tmp_path / "out.tif")
    with rasterio.open(tmp_path / "out.tif") as r:
        lab = r.read(1)
        assert r.nodata == 255
    assert (lab[:8, :8] == 255).all() and (lab[16:, 16:] == 1).all()


def test_streaming_infer_passes_normalise_through(tmp_path):
    _scene(tmp_path / "in.tif")
    res = StreamingInference(_Const(1), chip_size=32, overlap=0, device="cpu").infer(
        tmp_path / "in.tif", tmp_path / "out.tif", normalise=False)
    assert res["success"] and res["n_chips"] == 4
