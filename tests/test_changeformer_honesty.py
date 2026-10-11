"""The 'ChangeFormer' entries must describe what they build."""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.models.registry import get_model, get_model_spec


def test_b0_params_match_the_registry():
    model = get_model("changeformer-mit-b0", num_classes=2, in_channels=4, pretrained=False)
    measured = sum(p.numel() for p in model.parameters()) / 1e6
    assert abs(measured - get_model_spec("changeformer-mit-b0").params_m) < 0.2


def test_b4_is_refused_not_mislabelled():
    with pytest.raises(NotImplementedError, match="mislabelling"):
        get_model("changeformer-mit-b4", pretrained=False)


def test_unknown_backbone_label_is_refused():
    from pygeovision.models.change_detection.changeformer import ChangeFormer
    with pytest.raises(ValueError, match="single fixed architecture"):
        ChangeFormer(backbone="mit-b4")
    ChangeFormer(backbone="mit-b0")  # the legacy label still works


def test_stacked_input_matches_two_image_input():
    """[before | after] stacked must give the same result as forward(before, after)."""
    model = get_model("changeformer-mit-b0", num_classes=2, in_channels=4, pretrained=False).eval()
    before, after = torch.rand(1, 4, 32, 32), torch.rand(1, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(before, after), model(torch.cat([before, after], 1)), atol=1e-5)
