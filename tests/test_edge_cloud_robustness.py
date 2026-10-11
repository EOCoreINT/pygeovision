"""ONNX Runtime wrapper, Jetson converter and cloud deployers: behaviour that can be checked offline."""
from __future__ import annotations

import importlib.util
import logging
import sys
import tarfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
rasterio = pytest.importorskip("rasterio")

import torch.nn as nn  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pygeovision.cloud.deploy import (  # noqa: E402
    SAGEMAKER_INFERENCE_PY,
    AWSDeployer,
    AzureDeployer,
    build_sagemaker_archive,
)
from pygeovision.edge import JetsonDeployer, ONNXRuntimeInference  # noqa: E402


class Net(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, nc, 1))

    def forward(self, x):
        return self.f(x)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    p = tmp_path_factory.mktemp("sc") / "scene.tif"
    arr = (np.random.default_rng(0).random((4, 150, 150)) * 255).astype("float32")
    with rasterio.open(p, "w", driver="GTiff", height=150, width=150, count=4, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(500000, 5300000, 10, 10)) as d:
        d.write(arr)
    return p


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    torch.manual_seed(0)
    net = Net().eval()
    fixed = ONNXRuntimeInference.from_pytorch(net, d / "fixed.onnx", input_shape=(1, 4, 64, 64), simplify=False)
    dyn = ONNXRuntimeInference.from_pytorch(net, d / "dyn.onnx", input_shape=(1, 4, 64, 64), simplify=False,
                                            dynamic_spatial=True)
    return net, fixed, dyn, d


class TestONNXRuntime:
    def test_parity_with_pytorch(self, models):
        net, fixed, _, _ = models
        x = torch.rand(3, 4, 64, 64)
        np.testing.assert_allclose(fixed.infer(x.numpy()), net(x).detach().numpy(), atol=1e-5)

    def test_dynamic_spatial_accepts_other_sizes(self, models):
        net, fixed, dyn, _ = models
        for size in (32, 100):
            x = torch.rand(2, 4, size, size)
            np.testing.assert_allclose(dyn.infer(x.numpy()), net(x).detach().numpy(), atol=1e-5)
        with pytest.raises(Exception, match="dimensions"):
            fixed.infer(np.random.rand(1, 4, 32, 32))

    def test_fixed_export_still_tiles_by_adopting_its_size(self, models, scene, tmp_path, caplog):
        _, fixed, _, _ = models
        with caplog.at_level(logging.WARNING):
            r = fixed.infer_geotiff(scene, tmp_path / "p.tif", chip_size=128, overlap=32)
        assert r["success"], r
        assert "exported for 64x64" in caplog.text
        with rasterio.open(tmp_path / "p.tif") as p:
            assert (p.height, p.width) == (150, 150)

    def test_dynamic_export_uses_requested_chip_size(self, models, scene, tmp_path):
        _, _, dyn, _ = models
        assert dyn.infer_geotiff(scene, tmp_path / "p.tif", chip_size=100, overlap=20)["success"]

    def test_num_classes_comes_from_the_model(self, models, scene, tmp_path):
        _, _, dyn, _ = models
        dyn.infer_geotiff(scene, tmp_path / "p.tif", chip_size=64, overlap=16)
        with rasterio.open(tmp_path / "p.tif") as p:
            assert p.read(1).max() <= 5            # six classes, not the old default of two

    def test_wrong_num_classes_is_a_clear_error_not_a_numpy_crash(self, models, scene, tmp_path):
        _, _, dyn, _ = models
        r = dyn.infer_geotiff(scene, tmp_path / "p.tif", chip_size=64, overlap=16, num_classes=2)
        assert r["success"] is False and "outputs 6 channels" in r["error"]

    def test_wrong_band_count_is_reported(self, models, tmp_path):
        _, _, dyn, _ = models
        p = tmp_path / "three.tif"
        with rasterio.open(p, "w", driver="GTiff", height=80, width=80, count=3, dtype="float32",
                           crs="EPSG:32633", transform=from_origin(0, 0, 10, 10)) as d:
            d.write(np.random.rand(3, 80, 80).astype("float32"))
        r = dyn.infer_geotiff(p, tmp_path / "o.tif", chip_size=64, overlap=16)
        assert r["success"] is False and "3" in r["error"]

    def test_non_square_fixed_input_is_refused_clearly(self, tmp_path, scene):
        eng = ONNXRuntimeInference.from_pytorch(Net().eval(), tmp_path / "ns.onnx",
                                                input_shape=(1, 4, 64, 96), simplify=False)
        r = eng.infer_geotiff(scene, tmp_path / "o.tif")
        assert not r["success"] and "non-square" in r["error"]

    def test_benchmark_fields_and_validation(self, models):
        _, _, dyn, _ = models
        b = dyn.benchmark(n_runs=5)
        assert {"mean_ms", "median_ms", "p95_ms", "fps", "providers"} <= set(b)
        assert dyn.benchmark(n_runs=1)["n_runs"] == 1
        with pytest.raises(ValueError, match="at least 1"):
            dyn.benchmark(n_runs=0)

    def test_unavailable_gpu_is_reported_not_silent(self, models, caplog):
        _, _, _, d = models
        with caplog.at_level(logging.WARNING):
            ONNXRuntimeInference(d / "fixed.onnx", device="cuda")._load()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            assert "running on the CPU" in caplog.text


class TestTiledPreflightUsesChipSize:
    def test_error_names_the_chip_size(self, tmp_path, scene):
        from pygeovision.inference.tiled import TiledInference

        class Only64(nn.Module):
            def forward(self, x):
                if x.shape[-1] != 64:
                    raise RuntimeError("Got invalid dimensions for input")
                return torch.zeros(x.shape[0], 2, 64, 64)

        r = TiledInference(Only64(), chip_size=64, overlap=0, num_classes=2).infer(scene, tmp_path / "a.tif")
        assert r["success"], r                                  # a probe at 32x32 used to reject this
        r = TiledInference(Only64(), chip_size=48, overlap=0, num_classes=2).infer(scene, tmp_path / "b.tif")
        assert not r["success"] and "chip_size=48" in r["error"]


class TestJetson:
    def test_without_tensorrt_you_get_onnx_and_a_command(self, tmp_path):
        with patch.dict(sys.modules, {"tensorrt": None}):
            r = JetsonDeployer().convert(Net().eval(), tmp_path / "m.onnx", tmp_path / "m.trt",
                                         precision="fp16", input_shape=(1, 4, 64, 64))
        assert r["success"] is False and r["stage"] == "tensorrt_not_installed"
        assert "--fp16" in r["jetson_command"] and "--memPoolSize=workspace:4096M" in r["jetson_command"]
        assert "--workspace=" not in r["jetson_command"] and "noFP16" not in r["jetson_command"]
        assert (tmp_path / "m.onnx").exists()

    def test_fp32_command_has_no_precision_flag(self, tmp_path):
        with patch.dict(sys.modules, {"tensorrt": None}):
            r = JetsonDeployer().convert(Net().eval(), tmp_path / "m.onnx", tmp_path / "m.trt",
                                         precision="fp32", input_shape=(1, 4, 64, 64))
        assert "--fp16" not in r["jetson_command"] and "--int8" not in r["jetson_command"]

    def test_exported_onnx_has_static_shapes(self, tmp_path):
        import onnx
        with patch.dict(sys.modules, {"tensorrt": None}):
            JetsonDeployer().convert(Net().eval(), tmp_path / "m.onnx", tmp_path / "m.trt",
                                     input_shape=(1, 4, 64, 64))
        g = onnx.load(str(tmp_path / "m.onnx"))
        dims = [d.dim_value for d in g.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, 4, 64, 64]             # TensorRT needs a profile for anything dynamic

    def test_unknown_precision_raises(self, tmp_path):
        with pytest.raises(ValueError, match="precision"):
            JetsonDeployer().convert(Net(), tmp_path / "a.onnx", tmp_path / "a.trt", precision="fp8")

    def test_benchmark_without_tensorrt_is_an_error_dict(self):
        with patch.dict(sys.modules, {"tensorrt": None}):
            assert "TensorRT" in JetsonDeployer().benchmark("x.trt")["error"]

    def test_output_buffer_is_sized_from_the_engine(self):
        trt = types.SimpleNamespace(TensorIOMode=types.SimpleNamespace(OUTPUT="out"))
        engine = MagicMock()
        engine.num_io_tensors = 2
        engine.get_tensor_name.side_effect = ["input", "output"]
        engine.get_tensor_mode.side_effect = lambda n: "in" if n == "input" else "out"
        engine.get_tensor_shape.return_value = (-1, 6, 64, 64)
        assert JetsonDeployer._output_numel(engine, trt, batch=2) == 2 * 6 * 64 * 64
        engine.get_tensor_name.side_effect = ["input", "output"]
        engine.get_tensor_shape.return_value = (1, -1, 64, 64)
        with pytest.raises(ValueError, match="dynamic"):
            JetsonDeployer._output_numel(engine, trt, batch=1)


def _load_handler(path: Path):
    spec = importlib.util.spec_from_file_location("sm_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSageMakerArchive:
    def test_layout(self, models, tmp_path):
        _, _, dyn, d = models
        out = build_sagemaker_archive(d / "dyn.onnx", tmp_path / "m.tar.gz")
        with tarfile.open(out) as t:
            names = set(t.getnames())
        assert {"dyn.onnx", "code", "code/inference.py", "code/requirements.txt"} <= names

    def test_default_output_next_to_model(self, models):
        _, _, _, d = models
        assert build_sagemaker_archive(d / "fixed.onnx") == d / "fixed.tar.gz"

    def test_handler_serves_the_model_end_to_end(self, models, tmp_path):
        net, _, _, d = models
        out = build_sagemaker_archive(d / "dyn.onnx", tmp_path / "m.tar.gz")
        with tarfile.open(out) as t:
            t.extractall(tmp_path / "x")
        h = _load_handler(tmp_path / "x" / "code" / "inference.py")
        session = h.model_fn(str(tmp_path / "x"))
        x = torch.rand(4, 40, 40)
        data = h.input_fn(__import__("json").dumps({"inputs": x.tolist()}))
        assert data[0].shape == (1, 4, 40, 40) and data[1] is False   # a single chip gets a batch axis
        body, ctype = h.output_fn(h.predict_fn(data, session))
        pred = np.asarray(__import__("json").loads(body)["predictions"])
        np.testing.assert_allclose(pred, net(x[None]).detach().numpy(), atol=1e-5)
        assert ctype == "application/json"
        # "argmax": class ids instead of logits, much smaller on the wire
        small, _ = h.output_fn(h.predict_fn(h.input_fn(__import__("json").dumps({"inputs": x.tolist(), "argmax": True})), session))
        classes = np.asarray(__import__("json").loads(small)["predictions"])
        assert classes.shape == (1, 40, 40) and (classes == pred.argmax(1)).all()
        assert len(small) * 5 < len(body)
        with pytest.raises(ValueError, match="content type"):
            h.input_fn("[]", "text/csv")

    def test_handler_text_is_what_is_packed(self, models, tmp_path):
        _, _, _, d = models
        out = build_sagemaker_archive(d / "dyn.onnx", tmp_path / "m.tar.gz")
        with tarfile.open(out) as t:
            assert t.extractfile("code/inference.py").read().decode() == SAGEMAKER_INFERENCE_PY

    def test_rejects_non_onnx_and_missing(self, tmp_path):
        (tmp_path / "w.pt").write_bytes(b"x")
        with pytest.raises(ValueError, match="onnx"):
            build_sagemaker_archive(tmp_path / "w.pt")
        with pytest.raises(FileNotFoundError):
            build_sagemaker_archive(tmp_path / "none.onnx")


class TestAWSDeploy:
    @staticmethod
    def _sdk():
        boto3, sagemaker = MagicMock(), MagicMock()
        return {"boto3": boto3, "sagemaker": sagemaker, "sagemaker.model": sagemaker.model}, boto3

    def test_missing_bucket_and_role_are_named(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
        monkeypatch.delenv("AWS_SAGEMAKER_ROLE_ARN", raising=False)
        (tmp_path / "m.onnx").write_bytes(b"x")
        mods, _ = self._sdk()
        with patch.dict(sys.modules, mods):
            r = AWSDeployer().deploy(tmp_path / "m.onnx", "ep")
        assert r["success"] is False and "AWS_S3_BUCKET" in r["error"] and "AWS_SAGEMAKER_ROLE_ARN" in r["error"]

    def test_onnx_is_packed_before_upload(self, models, tmp_path):
        _, _, _, d = models
        mods, boto3 = self._sdk()
        with patch.dict(sys.modules, mods):
            r = AWSDeployer(bucket="b", role_arn="arn:aws:iam::1:role/r").deploy(
                d / "dyn.onnx", "ep", auto_scale=False)
        assert r["success"] is True, r
        _, bucket, key = boto3.client.return_value.upload_file.call_args[0]
        assert bucket == "b" and key.endswith(".tar.gz")
        assert r["model_s3_uri"].endswith(".tar.gz")

    def test_ready_archive_is_uploaded_unchanged(self, tmp_path):
        archive = tmp_path / "mine.tar.gz"
        archive.write_bytes(b"x")
        mods, boto3 = self._sdk()
        with patch.dict(sys.modules, mods):
            AWSDeployer(bucket="b", role_arn="r").deploy(archive, "ep", auto_scale=False)
        assert boto3.client.return_value.upload_file.call_args[0][0] == str(archive)

    def test_missing_model_file(self, tmp_path):
        mods, _ = self._sdk()
        with patch.dict(sys.modules, mods):
            r = AWSDeployer(bucket="b", role_arn="r").deploy(tmp_path / "nope.onnx", "ep")
        assert r["success"] is False and "not found" in r["error"]

    def test_listing_failure_is_logged_not_silent(self, caplog):
        dep = AWSDeployer()
        with patch.object(dep, "_get_client", side_effect=RuntimeError("no credentials")):
            with caplog.at_level(logging.WARNING):
                assert dep.list_endpoints() == []
        assert "no credentials" in caplog.text and "does NOT mean" in caplog.text


class TestAWSPredict:
    def test_array_is_sent_with_the_argmax_flag(self):
        import json as _json
        boto3 = MagicMock()
        runtime = boto3.client.return_value
        runtime.invoke_endpoint.return_value = {"Body": MagicMock(read=lambda: b'{"predictions": [[0]]}')}
        with patch.dict(sys.modules, {"boto3": boto3}):
            out = AWSDeployer().predict("ep", np.zeros((1, 2, 2)), argmax=True)
        assert out == {"predictions": [[0]]}
        sent = _json.loads(runtime.invoke_endpoint.call_args.kwargs["Body"])
        assert sent == {"inputs": [[[0.0, 0.0], [0.0, 0.0]]], "argmax": True}


class TestGCPDeploy:
    def test_each_endpoint_gets_its_own_artifact_folder(self, tmp_path):
        from pygeovision.cloud.deploy import GCPDeployer
        (tmp_path / "m.onnx").write_bytes(b"x")
        google = MagicMock()
        mods = {"google": google, "google.cloud": google.cloud,
                "google.cloud.aiplatform": google.cloud.aiplatform, "google.cloud.storage": google.cloud.storage}
        with patch.dict(sys.modules, mods):
            r = GCPDeployer(project_id="p").deploy(tmp_path / "m.onnx", "ep-a", gcs_bucket="b")
        assert r["success"] is True, r
        assert r["model_gcs_uri"] == "gs://b/pygeovision/models/ep-a/"


class TestIamPolicy:
    def test_policy_covers_the_calls_deploy_makes(self):
        import json as _json
        pol = _json.loads(AWSDeployer(bucket="b", role_arn="arn:aws:iam::1:role/r").generate_iam_policy())
        actions = {a for st in pol["Statement"] for a in ([st["Action"]] if isinstance(st["Action"], str) else st["Action"])}
        for needed in ("sagemaker:DescribeEndpoint", "sagemaker:CreateEndpointConfig", "iam:PassRole",
                       "application-autoscaling:PutScalingPolicy", "s3:PutObject"):
            assert needed in actions


class TestAzurePredict:
    def test_request_is_passed_as_a_file(self):
        seen = {}

        def invoke(endpoint_name, request_file):
            seen["path"] = request_file
            seen["body"] = Path(request_file).read_text()
            return "ok"

        ml = MagicMock()
        ml.online_endpoints.invoke.side_effect = invoke
        mods = {"azure": MagicMock(), "azure.ai": MagicMock(), "azure.ai.ml": MagicMock(MLClient=MagicMock(return_value=ml)),
                "azure.identity": MagicMock()}
        with patch.dict(sys.modules, mods):
            out = AzureDeployer(subscription_id="s", resource_group="g", workspace_name="w").predict(
                "ep", np.zeros((1, 2)))
        assert out == "ok" and seen["body"] == '{"inputs": [[0.0, 0.0]]}'
        assert not Path(seen["path"]).exists()          # the temporary file is cleaned up


class TestModelOptimizerExport:
    def test_spatial_size_is_fixed_unless_asked(self, tmp_path):
        from pygeovision.training.hpo import ModelOptimizer
        net = Net(3).eval()
        mo = ModelOptimizer(net, num_classes=3, in_channels=4)
        fixed = ONNXRuntimeInference(mo.export_onnx(str(tmp_path / "f.onnx"), input_size=32, simplify=False))
        dyn = ONNXRuntimeInference(mo.export_onnx(str(tmp_path / "d.onnx"), input_size=32, simplify=False,
                                                  dynamic_spatial=True))
        x = np.random.rand(1, 4, 48, 48).astype("float32")
        with pytest.raises(Exception, match="dimensions"):
            fixed.infer(x)
        np.testing.assert_allclose(dyn.infer(x), net(torch.tensor(x)).detach().numpy(), atol=1e-5)
