"""FocalLoss per-class alpha and WeightedSumLoss."""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from pygeovision.losses import DiceLoss, FocalLoss, WeightedSumLoss  # noqa: E402


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 2, 8, 8, generator=g)
    target = (torch.rand(2, 8, 8, generator=g) > 0.8).long()
    return logits, target


def reference_focal(logits, target, alpha_t, gamma=2.0):
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return (alpha_t * (1 - pt) ** gamma * ce).sum() / target.numel()


class TestFocalAlpha:
    def test_scalar_alpha_is_unchanged(self):
        logits, target = batch()
        assert FocalLoss(alpha=0.25)(logits, target).item() == pytest.approx(
            reference_focal(logits, target, 0.25).item(), rel=1e-6)

    def test_per_class_alpha_weights_each_pixel_by_its_class(self):
        logits, target = batch()
        alpha = torch.tensor([0.25, 0.75])[target]
        got = FocalLoss(alpha=[0.25, 0.75])(logits, target).item()
        assert got == pytest.approx(reference_focal(logits, target, alpha).item(), rel=1e-6)

    def test_rare_class_gets_more_weight_than_with_a_scalar(self):
        logits, target = batch()
        uniform = FocalLoss(alpha=0.25)(logits, target).item()
        balanced = FocalLoss(alpha=[0.25, 0.75])(logits, target).item()
        assert balanced > uniform

    def test_ignore_index_pixels_are_excluded(self):
        logits, target = batch()
        masked = target.clone()
        masked[:, :2] = 255
        v = FocalLoss(alpha=[0.25, 0.75])(logits, masked)
        assert torch.isfinite(v)

    def test_wrong_number_of_weights_is_an_error(self):
        logits, target = batch()
        with pytest.raises(ValueError, match="per-class weights"):
            FocalLoss(alpha=[0.25, 0.5, 0.25])(logits, target)


class TestWeightedSum:
    def test_sums_weighted_terms_and_backpropagates(self):
        logits, target = batch()
        logits.requires_grad_(True)
        focal, dice = FocalLoss(alpha=[0.25, 0.75]), DiceLoss()
        loss = WeightedSumLoss([(0.5, focal), (0.5, dice)])
        value = loss(logits, target)
        assert value.item() == pytest.approx(0.5 * focal(logits, target).item() + 0.5 * dice(logits, target).item(), rel=1e-6)
        value.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_needs_a_term(self):
        with pytest.raises(ValueError):
            WeightedSumLoss([])


class TestDiceClasses:
    def test_foreground_only_dice_is_not_diluted_by_the_easy_background(self):
        import torch

        from pygeovision.losses import DiceLoss
        # 10 of 100 pixels are class 1; the model predicts "all background" confidently
        tgt = torch.zeros(1, 10, 10, dtype=torch.long)
        tgt[0, 0, :] = 1
        logits = torch.zeros(1, 2, 10, 10)
        logits[:, 0] = 8.0                                   # certain: background
        both = float(DiceLoss()(logits, tgt))
        fg = float(DiceLoss(classes=[1])(logits, tgt))
        assert fg > 0.9                                      # missing the object is ~fully penalised
        assert both == pytest.approx((fg + float(DiceLoss(classes=[0])(logits, tgt))) / 2, abs=1e-5)
        assert fg > both * 1.5                               # the mean hides the miss

    def test_bad_class_index_is_refused(self):
        import torch

        from pygeovision.losses import DiceLoss
        with pytest.raises(ValueError, match="out of range"):
            DiceLoss(classes=[5])(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long))
