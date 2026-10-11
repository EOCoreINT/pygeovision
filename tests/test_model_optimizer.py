"""ModelOptimizer: ONNX export + ONNX Runtime, INT8 quantisation, pruning (CPU, tiny conv net)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from pygeovision.training.hpo import ModelOptimizer  # noqa: E402


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 2, 1)).eval()


def test_onnx_export_and_runtime_match_pytorch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    net = _net()
    mo = ModelOptimizer(net, num_classes=2, in_channels=4)
    path = mo.export_onnx(str(tmp_path / "m.onnx"), input_size=32, simplify=False)
    x = np.random.rand(4, 32, 32).astype("float32")
    out = mo.onnx_inference(x, path)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)[None]).numpy()
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-4)


def test_int8_quantisation_runs_and_stays_close(tmp_path):
    net = _net()
    x = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        ref = net(x)
    q = ModelOptimizer(net, in_channels=4).quantize_int8([(x, None), (torch.rand(2, 4, 32, 32), None)],
                                                          str(tmp_path / "q.pth"))
    with torch.no_grad():
        out = q(x)
    assert out.shape == ref.shape
    assert (tmp_path / "q.pth").exists()
    assert float((out - ref).abs().mean()) < 0.1


def test_int8_needs_calibration_data(tmp_path):
    with pytest.raises(ValueError, match="no batches"):
        ModelOptimizer(_net(), in_channels=4).quantize_int8([], str(tmp_path / "q.pth"))


def test_pruning_zeroes_weights_but_keeps_plain_state_dict_keys():
    net = _net()
    keys = set(net.state_dict())
    ModelOptimizer(net, in_channels=4).prune(0.5)
    assert set(net.state_dict()) == keys                       # no weight_orig / weight_mask
    w = net[0].weight
    assert float((w == 0).float().mean()) == pytest.approx(0.5, abs=0.05)
    assert float((net[1].weight == 0).float().mean()) == 0.0   # batch-norm scales are untouched


def test_unknown_prune_method_is_refused():
    with pytest.raises(ValueError, match="prune method"):
        ModelOptimizer(_net(), in_channels=4).prune(0.3, method="random")
