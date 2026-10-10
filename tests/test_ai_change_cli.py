"""``client.change.detect`` / ``pygeovision ai change``: no silent substitution, honest spectral method."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin


def _write(path, arr, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:32633", transform=from_origin(500000, 5000000, 10, 10),
                       nodata=nodata) as dst:
        dst.write(arr.astype("float32"))


@pytest.fixture()
def pair(tmp_path):
    rng = np.random.default_rng(0)
    a = 0.2 + 0.01 * rng.standard_normal((3, 64, 64))
    b = a.copy()
    b[:, 10:30, 10:30] += 0.5            # a real, spatially compact change: 400 px of 4096
    pa, pb = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(pa, a); _write(pb, b)
    return pa, pb


def _detect(*a, **k):
    from pygeovision import PyGeoVision
    return PyGeoVision().change.detect(*a, **k)


def test_spectral_diff_finds_the_changed_block_not_a_fixed_percentile(pair, tmp_path):
    res = _detect(str(pair[0]), str(pair[1]), output_path=str(tmp_path / "o.tif"))
    assert res["method"] == "spectral_diff"
    assert res["change_fraction"] == pytest.approx(400 / 4096, abs=0.01)    # the old code always said 0.10
    with rasterio.open(tmp_path / "o.tif") as src:
        m = src.read(1)
    assert m[10:30, 10:30].all() and not m[40:, 40:].any()


def test_unchanged_scene_with_floor_reports_zero_change(pair, tmp_path):
    res = _detect(str(pair[0]), str(pair[0]), output_path=str(tmp_path / "o.tif"), min_magnitude=0.05)
    assert res["change_fraction"] == 0.0


def test_nodata_pixels_are_255_and_excluded(tmp_path):
    a = np.full((2, 20, 20), 0.3); b = a.copy(); b[:, :10] += 0.5
    a[:, 0, :] = -9999
    pa, pb = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(pa, a, nodata=-9999); _write(pb, b, nodata=-9999)
    res = _detect(str(pa), str(pb), output_path=str(tmp_path / "o.tif"))
    with rasterio.open(tmp_path / "o.tif") as src:
        m = src.read(1)
        assert src.nodata == 255
    assert (m[0] == 255).all() and res["valid_fraction"] == pytest.approx(0.95)


@pytest.mark.parametrize("method", ["changeformer", "bit", "dsamnet"])
def test_network_methods_refuse_without_checkpoint(pair, tmp_path, method):
    with pytest.raises(ValueError, match="checkpoint"):
        _detect(str(pair[0]), str(pair[1]), output_path=str(tmp_path / "o.tif"), method=method)
    assert not (tmp_path / "o.tif").exists()      # no spectral-diff output was written in its place


def test_mismatched_checkpoint_fails_loudly(pair, tmp_path):
    torch = pytest.importorskip("torch")
    ck = tmp_path / "bad.pth"
    torch.save({"conv.weight": torch.zeros(1)}, ck)
    with pytest.raises(RuntimeError):
        _detect(str(pair[0]), str(pair[1]), output_path=str(tmp_path / "o.tif"),
                method="changeformer", checkpoint=str(ck))


def test_checkpoint_roundtrip_runs_the_network(pair, tmp_path):
    torch = pytest.importorskip("torch")
    from pygeovision.models.change_detection.changeformer import ChangeDetection
    net = ChangeDetection(model_variant="changeformer", num_classes=2, in_channels=3).build()
    ck = tmp_path / "ok.pth"
    torch.save({"state_dict": net.state_dict()}, ck)
    res = _detect(str(pair[0]), str(pair[1]), output_path=str(tmp_path / "o.tif"),
                  method="changeformer", checkpoint=str(ck))
    assert res["model"] == "ChangeFormer" and (tmp_path / "o.tif").exists()


def test_cli_default_and_error_exit_codes(pair, tmp_path):
    from pygeovision.cli.main import cli
    r = CliRunner().invoke(cli, ["ai", "change", "--before", str(pair[0]), "--after", str(pair[1]),
                                 "-o", str(tmp_path / "o.tif")])
    assert r.exit_code == 0, r.output
    assert "change_fraction" in r.output
    r = CliRunner().invoke(cli, ["ai", "change", "--before", str(pair[0]), "--after", str(pair[1]),
                                 "-o", str(tmp_path / "p.tif"), "--method", "bit"])
    assert r.exit_code != 0 and "checkpoint" in r.output and not (tmp_path / "p.tif").exists()
    r = CliRunner().invoke(cli, ["ai", "change", "--before", str(pair[0]), "--after", str(pair[1]),
                                 "-o", str(tmp_path / "q.tif"), "--min-magnitude", "1", "--method", "bit"])
    assert r.exit_code != 0
