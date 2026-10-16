"""The deployment scripts printed in the docs (docs/_examples/deploy) must actually serve a model."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

EX = Path(__file__).resolve().parents[1] / "docs" / "_examples" / "deploy"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def onnx_model(tmp_path):
    net = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 1)).eval()
    path = tmp_path / "models" / "tiny.onnx"
    path.parent.mkdir()
    torch.onnx.export(net, torch.zeros(1, 4, 16, 16), str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}, "logits": {0: "b", 2: "h", 3: "w"}}, dynamo=False)
    return path, net


def test_azure_score_script_matches_pytorch(onnx_model, monkeypatch):
    path, net = onnx_model
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(path.parent))
    score = _load("azure_score", EX / "azure_score.py")
    score.init()
    chip = np.random.default_rng(0).random((4, 16, 16)).astype("float32")
    out = score.run(json.dumps({"inputs": chip.tolist()}))
    expected = net(torch.from_numpy(chip)[None]).detach().numpy()
    assert np.allclose(np.asarray(out["predictions"]), expected, atol=1e-5)
    ids = score.run(json.dumps({"inputs": chip.tolist(), "argmax": True}))
    assert np.asarray(ids["predictions"]).shape == (1, 16, 16)


def test_vertex_server_serves_health_and_predict(onnx_model, monkeypatch):
    path, net = onnx_model
    monkeypatch.setenv("MODEL_PATH", str(path))
    server_mod = _load("vertex_server", EX / "vertex_server.py")
    httpd = server_mod.HTTPServer(("127.0.0.1", 0), server_mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
        chip = np.random.default_rng(1).random((4, 16, 16)).astype("float32")
        req = urllib.request.Request(base + "/predict", data=json.dumps({"instances": [chip.tolist()],
                                     "parameters": {"argmax": True}}).encode(), method="POST")
        got = json.load(urllib.request.urlopen(req))["predictions"]
        expected = net(torch.from_numpy(chip)[None]).argmax(1).numpy()
        assert np.array_equal(np.asarray(got), expected)
        bad = urllib.request.Request(base + "/predict", data=b'{"nope": 1}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        httpd.shutdown()
