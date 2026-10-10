"""The YAML pipeline engine does real work or fails: no 'N steps completed' for steps that did nothing."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import yaml
from click.testing import CliRunner
from rasterio.transform import from_origin

from pygeovision.cli.main import cli
from pygeovision.pipelines.orchestrator import Pipeline
from pygeovision.pipelines.steps import Step, StepError
from pygeovision.pipelines.yaml_parser import PipelineYAMLParser

torch = pytest.importorskip("torch")
BANDS, CLASSES = 3, 2


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from pygeovision.ai.models.hub import ModelHub
    m = ModelHub().load("unet_resnet50", num_classes=CLASSES, in_channels=BANDS, pretrained=False)
    p = tmp_path_factory.mktemp("w") / "m.pth"
    torch.save(m.state_dict(), p)
    return p


@pytest.fixture()
def scene(tmp_path):
    p = tmp_path / "scene.tif"
    with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=BANDS, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(500000, 5000000, 10, 10)) as d:
        d.write(np.random.default_rng(1).random((BANDS, 64, 64)).astype("float32"))
    return p


class FakeClient:
    def __init__(self, scene, found=True, downloaded=True):
        self.scene, self.found, self.downloaded = scene, found, downloaded
        self.search_kwargs = None

    def search(self, **kw):
        self.search_kwargs = kw
        return [SimpleNamespace(id="s1")] if self.found else []

    def download(self, items, **kw):
        if not self.downloaded:
            return [SimpleNamespace(success=False, path=None, error="403")]
        return [SimpleNamespace(success=True, path=self.scene, error="")]


def config(tmp_path, ckpt_path=None, fmt="geojson"):
    infer = {"model": "unet_resnet50", "num_classes": CLASSES, "chip_size": 32, "overlap": 8,
             "output_dir": str(tmp_path / "pred")}
    if ckpt_path:
        infer["checkpoint"] = str(ckpt_path)
    return {"name": "t", "steps": [
        {"name": "search", "action": "search", "params": {"bbox": [0, 0, 1, 1], "date_range": ["2024-01-01", "2024-02-01"]}},
        {"name": "download", "action": "download", "depends_on": ["search"], "params": {}},
        {"name": "infer", "action": "infer", "depends_on": ["download"], "params": infer},
        {"name": "export", "action": "export", "depends_on": ["infer"],
         "params": {"format": fmt, "output_dir": str(tmp_path / "vec")}}]}


def test_full_run_writes_prediction_and_vectors(tmp_path, scene, ckpt):
    res = Pipeline.from_config(config(tmp_path, ckpt)).run({"client": FakeClient(scene)})
    assert res.success and res.steps_completed == ["search", "download", "infer", "export"], res.error
    assert (tmp_path / "pred" / "scene_pred.tif").exists()
    out = tmp_path / "vec" / "scene_pred.geojson"
    assert out.exists() and json.loads(out.read_text())["type"] == "FeatureCollection"


def test_gpkg_export_format(tmp_path, scene, ckpt):
    res = Pipeline.from_config(config(tmp_path, ckpt, fmt="gpkg")).run({"client": FakeClient(scene)})
    assert res.success and (tmp_path / "vec" / "scene_pred.gpkg").exists()
    assert not (tmp_path / "vec" / "scene_pred.geojson").exists()


def test_infer_without_checkpoint_fails_instead_of_inferring_with_random_weights(tmp_path, scene):
    res = Pipeline.from_config(config(tmp_path)).run({"client": FakeClient(scene)})
    assert not res.success and res.steps_failed == ["infer", "export"]
    assert "checkpoint" in res.error and not (tmp_path / "pred").exists()


def test_no_client_empty_search_and_failed_download_are_failures(tmp_path, scene, ckpt):
    assert not Pipeline.from_config(config(tmp_path, ckpt)).run({}).success                    # was a "success"
    r = Pipeline.from_config(config(tmp_path, ckpt)).run({"client": FakeClient(scene, found=False)})
    assert not r.success and "no scenes" in r.error
    r = Pipeline.from_config(config(tmp_path, ckpt)).run({"client": FakeClient(scene, downloaded=False)})
    assert not r.success and "403" in r.error


def test_step_errors_are_not_retried(tmp_path, scene, monkeypatch):
    import time
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    cfg = config(tmp_path)
    cfg["steps"][2]["retry_on_fail"] = 5
    Pipeline.from_config(cfg).run({"client": FakeClient(scene)})
    assert slept == []


def test_unimplemented_actions_are_rejected_not_no_ops(tmp_path):
    cfg = {"name": "x", "steps": [{"name": "a", "action": "notify"}]}
    with pytest.raises(ValueError, match="notify"):
        PipelineYAMLParser().validate(cfg)
    with pytest.raises(StepError, match="no built-in implementation"):
        Step(name="a", action="custom").run({})
    assert Step(name="a", action="custom", _fn=lambda ctx, **k: {"ok": 1}).run({}) == {"ok": 1}


def test_overrides(tmp_path):
    cfg = config(tmp_path)
    PipelineYAMLParser.apply_overrides(cfg, ["search.bbox=[1,2,3,4]", "infer.checkpoint=best.pth"])
    assert cfg["steps"][0]["params"]["bbox"] == [1, 2, 3, 4]
    assert cfg["steps"][2]["params"]["checkpoint"] == "best.pth"
    with pytest.raises(ValueError, match="no step named"):
        PipelineYAMLParser.apply_overrides(cfg, ["nope.x=1"])
    with pytest.raises(ValueError, match="step.param=value"):
        PipelineYAMLParser.apply_overrides(cfg, ["bbox"])


def test_cli_run_and_templates(tmp_path, scene, ckpt, monkeypatch):
    import pygeovision
    fake = FakeClient(scene)
    monkeypatch.setattr(pygeovision.PyGeoVision, "search", lambda self, **kw: fake.search(**kw))
    monkeypatch.setattr(pygeovision.PyGeoVision, "download", lambda self, items, **kw: fake.download(items, **kw))
    cfg = config(tmp_path)
    y = tmp_path / "p.yaml"; y.write_text(yaml.safe_dump(cfg))
    r = CliRunner().invoke(cli, ["pipeline", "run", str(y)])
    assert r.exit_code == 1 and "checkpoint" in r.output                       # no weights -> not a success
    r = CliRunner().invoke(cli, ["pipeline", "run", str(y), "--set", f"infer.checkpoint={ckpt}"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "vec" / "scene_pred.geojson").exists()
    assert CliRunner().invoke(cli, ["pipeline", "run", str(y), "--dry-run"]).exit_code == 0
    t = CliRunner().invoke(cli, ["pipeline", "list-templates"])
    assert "urban" in t.output and "(0)" not in t.output


def test_shipped_templates_validate_and_need_user_input(tmp_path):
    from pathlib import Path

    import pygeovision.pipelines as pp
    files = sorted((Path(pp.__file__).parent / "templates").glob("*.yaml"))
    assert len(files) == 6
    for f in files:
        PipelineYAMLParser().load(f)                       # structurally valid
        res = Pipeline.from_yaml(str(f)).run({"client": FakeClient(None)})
        assert not res.success                             # no bbox/dates/checkpoint -> refuses, never "succeeds"
