"""Losses must refuse names they would otherwise ignore, and class weights must follow the logits."""
import pytest

torch = pytest.importorskip("torch")

from pygeovision.losses import BoundaryAwareLoss, ClassBalancedCrossEntropy, GeospatialMixedLoss  # noqa: E402


def test_mixed_loss_rejects_unknown_component():
    with pytest.raises(ValueError, match="unknown component"):
        GeospatialMixedLoss(weights={"combo": 0.5, "boundry": 0.5})


def test_mixed_loss_accepts_all_documented_components():
    logits, target = torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16))
    for name in GeospatialMixedLoss.COMPONENTS:
        assert torch.isfinite(GeospatialMixedLoss(weights={name: 1.0})(logits, target))


def test_boundary_loss_rejects_unknown_base():
    with pytest.raises(ValueError, match="base_loss"):
        BoundaryAwareLoss(base_loss="focal")


def test_class_balanced_ce_weights_the_rare_class_up():
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 0] = 2.0                      # favours class 0, so the single class-1 pixel is wrong
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, 0] = 1
    plain = torch.nn.functional.cross_entropy(logits, target)
    weighted = ClassBalancedCrossEntropy(class_counts=[1000, 10], beta=0.999)(logits, target)
    assert weighted.item() != pytest.approx(plain.item())
    # no counts -> ordinary cross-entropy
    assert ClassBalancedCrossEntropy()(logits, target).item() == pytest.approx(plain.item())


def test_tversky_classes_restricts_to_the_chosen_class():
    from pygeovision.losses import TverskyLoss

    logits = torch.randn(2, 3, 16, 16)
    target = torch.randint(0, 3, (2, 16, 16))
    only1 = TverskyLoss(classes=[1])(logits, target)
    every = TverskyLoss()(logits, target)
    assert only1.item() != pytest.approx(every.item())
    with pytest.raises(ValueError, match="out of range"):
        TverskyLoss(classes=[5])(logits, target)
