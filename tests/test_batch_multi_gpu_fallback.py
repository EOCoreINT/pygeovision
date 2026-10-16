"""BatchInferenceEngine.run_multi_gpu without a GPU must predict the files it was given, not the whole folder."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from pygeovision.inference import BatchInferenceEngine  # noqa: E402


def _write(path, seed):
    rng = np.random.default_rng(seed)
    kw = dict(driver="GTiff", height=64, width=64, count=4, dtype="float32", crs="EPSG:32633",
              transform=from_origin(500000, 5000000, 10, 10))
    with rasterio.open(path, "w", **kw) as dst:
        dst.write(rng.random((4, 64, 64)).astype("float32"))


def test_multi_gpu_fallback_predicts_only_the_requested_files(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        _write(src / f"s{i}.tif", i)
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1)).eval()
    engine = BatchInferenceEngine(model, device="cpu", chip_size=32, overlap=8, num_classes=2,
                                  normalise=False, half_precision=False)
    out = tmp_path / "out"
    res = engine.run_multi_gpu([str(src / "s0.tif"), str(src / "s2.tif")], str(out))
    assert sorted(p.name for p in out.glob("*.tif")) == ["s0.tif", "s2.tif"]        # s1 was not requested
    assert len(res["succeeded"]) == 2 and res["failed"] == []
