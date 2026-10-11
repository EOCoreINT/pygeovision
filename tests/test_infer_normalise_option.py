"""`--normalise/--no-normalise` on the inference commands, and honest batch failure reporting."""
import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from click.testing import CliRunner  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.cli.main import cli  # noqa: E402
from pygeovision.inference.batch import BatchInferenceEngine  # noqa: E402


class ThresholdNet(torch.nn.Module):
    """Class 1 where band 0 is above 0.5 (so it only works on unstretched 0-1 data)."""

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        v = x[:, :1]
        return torch.cat([0.5 - v, v - 0.5], dim=1) * 10


def _scene(path, bands=4, size=64):
    arr = np.full((bands, size, size), 0.2, "float32")
    arr[:, :, size // 2:] = 0.8
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands, dtype="float32",
                       crs="EPSG:32630", transform=from_origin(500000, 700000, 10, 10)) as d:
        d.write(arr)


def _fraction_class1(path):
    with rasterio.open(path) as r:
        return float((r.read(1) == 1).mean())


def test_no_normalise_changes_what_the_model_sees(tmp_path, monkeypatch):
    import importlib

    m = importlib.import_module("pygeovision.cli.main")

    monkeypatch.setattr(m, "_load_user_model", lambda *a, **k: ThresholdNet())
    img = tmp_path / "s.tif"
    _scene(img)
    ck = tmp_path / "c.pth"
    torch.save({}, ck)
    for flag, expect in (("--no-normalise", 0.5), ("--normalise", None)):
        out = tmp_path / f"o{flag}.tif"
        r = CliRunner().invoke(cli, ["infer", "predict", str(img), "-m", "x", "--checkpoint", str(ck),
                                     "-o", str(out), "--chip-size", "32", "--overlap", "0",
                                     "--device", "cpu", flag])
        assert r.exit_code == 0, r.output
        if expect is not None:
            assert _fraction_class1(out) == pytest.approx(expect, abs=0.02)
    # the stretch maps 0.2 -> 0 and 0.8 -> 1 so it is also half/half here; the point of the
    # option is the printed input description
    r = CliRunner().invoke(cli, ["infer", "predict", str(img), "-m", "x", "--checkpoint", str(ck),
                                 "-o", str(tmp_path / "z.tif"), "--chip-size", "32", "--overlap", "0",
                                 "--device", "cpu", "--no-normalise"])
    assert "used as stored" in r.output


def test_batch_reports_a_failed_file_as_failed(tmp_path):
    class NeedsFour(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 2, 1)

        def forward(self, x):
            return self.conv(x)

    (tmp_path / "in").mkdir()
    _scene(tmp_path / "in" / "ok.tif", bands=4)
    _scene(tmp_path / "in" / "bad.tif", bands=2)
    eng = BatchInferenceEngine(NeedsFour(), n_workers=1, chip_size=32, overlap=0, device="cpu",
                               half_precision=False)
    res = eng.run_directory(tmp_path / "in", tmp_path / "out")
    assert res["n_success"] + res["n_failed"] == 2
    assert res["n_failed"] >= 1
    assert any("bad.tif" in f["path"] for f in res["failed"])


def test_batch_cli_skips_existing_outputs_and_says_so(tmp_path, monkeypatch):
    import importlib

    m = importlib.import_module("pygeovision.cli.main")
    monkeypatch.setattr(m, "_load_user_model", lambda *a, **k: ThresholdNet())
    (tmp_path / "in").mkdir()
    _scene(tmp_path / "in" / "a.tif")
    ck = tmp_path / "c.pth"
    torch.save({}, ck)
    args = ["infer", "batch", str(tmp_path / "in"), str(tmp_path / "out"), "-m", "x",
            "--checkpoint", str(ck), "--no-normalise"]
    first = CliRunner().invoke(cli, args)
    assert first.exit_code == 0 and "1/1 succeeded" in first.output
    again = CliRunner().invoke(cli, args)
    assert again.exit_code == 0, again.output                 # nothing to do is not a failure
    assert "1 skipped" in again.output
    redo = CliRunner().invoke(cli, args + ["--overwrite"])
    assert "1/1 succeeded" in redo.output and "skipped" not in redo.output
