# ruff: noqa: E702, N802, C408, E731
"""SARDINOv3Adapter must not hand back zero 'features' as if they were real."""
import numpy as np
import pytest

from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter


def _no_backbone(monkeypatch):
    monkeypatch.setattr(SARDINOv3Adapter, "_load_model", lambda self: None)


def test_unloadable_backbone_raises_instead_of_returning_zeros(monkeypatch):
    _no_backbone(monkeypatch)
    vv = np.random.rand(56, 56).astype("float32"); vh = np.random.rand(56, 56).astype("float32")
    with pytest.raises(RuntimeError, match="could not be loaded"):
        SARDINOv3Adapter(model_size="vitb14").extract_features(vv, vh)


def test_allow_mock_returns_flagged_zero_arrays_of_the_right_shape(monkeypatch):
    _no_backbone(monkeypatch)
    vv = np.random.rand(56, 70).astype("float32"); vh = np.random.rand(56, 70).astype("float32")
    out = SARDINOv3Adapter(model_size="vitb14", allow_mock=True).extract_features(vv, vh)
    assert out["is_mock"] is True and out["cls_token"].shape == (768,) and out["patch_features"].shape == (4, 5, 768)
    assert not out["cls_token"].any() and out["normalised_input"].shape == (3, 56, 70)
