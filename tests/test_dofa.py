"""DOFA integration: validation of inputs, and no silent random-weight substitution (torchgeo is faked)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pygeovision.models.foundation import dofa  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        self.lin = torch.nn.Linear(2, 2)

    def forward_features(self, x, wavelengths):
        return torch.ones(1, 768) * x.shape[1]


@pytest.fixture()
def fake_torchgeo(monkeypatch):
    mod = types.ModuleType("torchgeo.models.dofa")
    for name in dofa.DOFA_MODELS:
        setattr(mod, name, lambda weights=None: _Net(weights))
    mod.DOFABase16_Weights = ["published-base"]
    mod.DOFALarge16_Weights = ["published-large"]
    pkg = types.ModuleType("torchgeo")
    models = types.ModuleType("torchgeo.models")
    models.dofa = mod
    pkg.models = models
    monkeypatch.setitem(sys.modules, "torchgeo", pkg)
    monkeypatch.setitem(sys.modules, "torchgeo.models", models)
    monkeypatch.setitem(sys.modules, "torchgeo.models.dofa", mod)


def test_models_without_published_weights_refuse_to_load_unless_allowed(fake_torchgeo):
    for name in ("dofa_small_patch16_224", "dofa_huge_patch14_224"):
        with pytest.raises(RuntimeError, match="No pretrained weights are published"):
            dofa.load_dofa_hf(name)
        net = dofa.load_dofa_hf(name, allow_random_init=True)
        assert net._pygeovision_is_random_init is True


def test_published_model_loads_with_its_weights_and_is_not_flagged(fake_torchgeo):
    net = dofa.load_dofa_hf("dofa_base_patch16_224")
    assert net.weights == "published-base"
    assert not getattr(net, "_pygeovision_is_random_init", False)


def test_unknown_model_name_is_a_value_error():
    with pytest.raises(ValueError, match="Unknown DOFA model"):
        dofa.load_dofa_hf("dofa_tiny")


def test_embedding_input_checks():
    net = _Net()
    img = np.zeros((9, 8, 8), "float32")
    with pytest.raises(ValueError, match="exactly one"):
        dofa.extract_dofa_embedding(net, img)
    with pytest.raises(ValueError, match="exactly one"):
        dofa.extract_dofa_embedding(net, img, sensor="sentinel2", wavelengths=[1.0] * 9)
    with pytest.raises(ValueError, match="No real, verified wavelength"):
        dofa.extract_dofa_embedding(net, img, sensor="landsat9")
    with pytest.raises(ValueError, match="6 real band"):
        dofa.extract_dofa_embedding(net, np.zeros((6, 8, 8), "float32"), sensor="sentinel2")


def test_channel_last_array_is_accepted():
    emb = dofa.extract_dofa_embedding(_Net(), np.zeros((16, 16, 9), "float32"), sensor="sentinel2")
    assert emb.shape == (768,) and emb[0] == 9
