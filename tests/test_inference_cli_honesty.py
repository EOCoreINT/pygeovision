"""Inference entry points run *your* trained network: strict checkpoint, no substitution, honest exit codes."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")

from pygeovision.ai.models.hub import ModelHub  # noqa: E402
from pygeovision.cli.main import cli  # noqa: E402
from pygeovision.core.exceptions import ModelLoadError  # noqa: E402

BANDS, CLASSES = 3, 4


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    m = ModelHub().load("unet_resnet50", num_classes=CLASSES, in_channels=BANDS, pretrained=False)
    p = tmp_path_factory.mktemp("w") / "unet.pth"
    torch.save({"state_dict": m.state_dict()}, p)
    return p


@pytest.fixture()
def scene(tmp_path):
    p = tmp_path / "s.tif"
    arr = np.random.default_rng(0).random((BANDS, 96, 96)).astype("float32")
    with rasterio.open(p, "w", driver="GTiff", height=96, width=96, count=BANDS, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(0, 960, 10, 10), nodata=-9999.0,
                       tiled=True, blockxsize=32, blockysize=32) as d:
        d.write(arr)
    return p


def _run(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


# ---- ModelHub.load -------------------------------------------------------------------------
def test_hub_does_not_substitute_by_default():
    with pytest.raises(ModelLoadError, match="Nothing was substituted"):
        ModelHub().load("no_such_model", num_classes=2, in_channels=3)


def test_hub_fallback_is_opt_in_and_flagged(monkeypatch):
    hub = ModelHub()
    real = hub._build_with_fallback

    def boom_first(name, kw, attempted, allow_fallback=True):
        if name == "unet_resnet50":
            from pygeovision.ai.models.registry import registry
            orig = registry.build

            def build(n, **k):
                if n == "unet_resnet50":
                    raise RuntimeError("offline")
                return orig(n, pretrained=False, **{x: v for x, v in k.items() if x != "pretrained"})
            monkeypatch.setattr(registry, "build", build)
            import pygeovision.models.registry as full
            monkeypatch.setattr(full, "get_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        return real(name, kw, attempted, allow_fallback=allow_fallback)

    monkeypatch.setattr(hub, "_build_with_fallback", boom_first)
    with pytest.raises(ModelLoadError):
        hub.load("unet_resnet50", num_classes=2, in_channels=3, pretrained=False)
    m = hub.load("unet_resnet50", num_classes=2, in_channels=3, pretrained=False, allow_fallback=True)
    assert m._pygeovision_fallback_used and m._pygeovision_actual_name != "unet_resnet50"


def test_hub_checkpoint_is_strict(ckpt):
    ok = ModelHub().load("unet_resnet50", checkpoint=str(ckpt), num_classes=CLASSES, in_channels=BANDS)
    assert ok is not None
    with pytest.raises(ModelLoadError, match="does not match"):
        ModelHub().load("unet_resnet50", checkpoint=str(ckpt), num_classes=CLASSES + 1, in_channels=BANDS)
    with pytest.raises(ModelLoadError, match="does not match"):
        ModelHub().load("unet_resnet50", checkpoint=str(ckpt), num_classes=CLASSES, in_channels=BANDS + 1)


# ---- CLI -----------------------------------------------------------------------------------
def test_infer_predict_requires_checkpoint(scene, tmp_path):
    r = _run("infer", "predict", scene, "-o", tmp_path / "o.tif")
    assert r.exit_code == 2 and "--checkpoint" in r.output and not (tmp_path / "o.tif").exists()


def test_infer_predict_runs_and_writes_valid_uint8(scene, ckpt, tmp_path):
    out = tmp_path / "o.tif"
    r = _run("infer", "predict", scene, "--checkpoint", ckpt, "-o", out, "-c", CLASSES,
             "--chip-size", 64, "--overlap", 16)
    assert r.exit_code == 0, r.output
    with rasterio.open(out) as src:
        a = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata is None      # -9999 does not fit uint8
    assert a.shape == (96, 96) and a.max() < CLASSES


@pytest.mark.parametrize("extra", [["-c", CLASSES + 1], ["-m", "nope"], ["--in-channels", BANDS + 1]])
def test_infer_predict_bad_checkpoint_or_model_exits_nonzero(scene, ckpt, tmp_path, extra):
    r = _run("infer", "predict", scene, "--checkpoint", ckpt, "-o", tmp_path / "o.tif",
             *([] if "-c" in extra else ["-c", CLASSES]), *extra)
    assert r.exit_code != 0 and not (tmp_path / "o.tif").exists()


def test_ai_infer_and_batch_and_custom(scene, ckpt, tmp_path):
    r = _run("ai", "infer", "-i", scene, "-m", "unet_resnet50", "--checkpoint", ckpt, "-o", tmp_path / "a.tif",
             "--num-classes", CLASSES, "--tile-size", 64, "--overlap", 16)
    assert r.exit_code == 0 and (tmp_path / "a.tif").exists(), r.output
    d = tmp_path / "in"; d.mkdir()
    (d / "x.tif").write_bytes(scene.read_bytes())
    r = _run("infer", "batch", d, tmp_path / "out", "--checkpoint", ckpt, "-c", CLASSES, "-w", 1)
    assert r.exit_code == 0 and "1/1" in r.output, r.output
    r = _run("ai", "segment", "custom", "-i", scene, "-o", tmp_path / "c.tif", "--model", "unet_resnet50")
    assert r.exit_code == 2 and "--checkpoint" in r.output          # custom without weights refuses


def test_client_custom_refuses_a_name_without_checkpoint(scene, tmp_path):
    from pygeovision import PyGeoVision
    with pytest.raises(ValueError, match="checkpoint"):
        PyGeoVision().segmentation.custom(str(scene), "unet_resnet50", output_path=str(tmp_path / "o.tif"))


def test_failure_paths_exit_nonzero(tmp_path):
    """These used to print a cross and exit 0, so scripts and CI could not see the failure."""
    r = _run("edge", "benchmark-onnx", tmp_path / "missing.onnx")
    assert r.exit_code != 0
    r = _run("edge", "export-onnx", "no-such-model", "-o", tmp_path / "m.onnx")
    assert r.exit_code != 0
    r = _run("pipeline", "run", tmp_path / "missing.yaml")
    assert r.exit_code != 0
