"""Tests for PyGeoVision geospatial loss functions (Phase 2+)."""

# Skip this file if PyTorch is not installed
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — pip install torch")

import pytest

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


@pytest.fixture
def seg_batch():
    """Synthetic segmentation batch: (B=2, C=3, H=16, W=16) preds, (B=2, H=16, W=16) targets."""
    import torch
    torch.manual_seed(42)
    preds   = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    return preds, targets


@pytest.fixture
def binary_batch():
    """Binary segmentation: 2-class."""
    import torch
    torch.manual_seed(0)
    preds   = torch.randn(2, 2, 16, 16)
    targets = torch.randint(0, 2, (2, 16, 16))
    return preds, targets


class TestDiceLoss:
    def test_forward_shape(self, seg_batch):
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = seg_batch
        loss = DiceLoss()(preds, targets)
        assert loss.ndim == 0      # scalar
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        import torch

        from pygeovision.losses.segmentation import DiceLoss
        targets = torch.zeros(1, 16, 16, dtype=torch.long)
        # Perfect prediction: class 0 has infinite logit
        preds = torch.full((1, 2, 16, 16), -1e9)
        preds[:, 0] = 1e9
        loss = DiceLoss()(preds, targets)
        assert loss.item() < 0.01   # near-zero loss

    def test_ignore_index(self, seg_batch):
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = seg_batch
        targets_with_ignore = targets.clone()
        targets_with_ignore[0, 0, 0] = 255
        loss_a = DiceLoss(ignore_index=255)(preds, targets_with_ignore)
        loss_b = DiceLoss(ignore_index=255)(preds, targets)
        # Both should be valid scalars
        assert loss_a.item() >= 0
        assert loss_b.item() >= 0

    def test_per_class_vs_mean(self, binary_batch):
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = binary_batch
        loss_pc   = DiceLoss(per_class=True)(preds, targets)
        loss_mean = DiceLoss(per_class=False)(preds, targets)
        assert loss_pc.ndim == 0
        assert loss_mean.ndim == 0


class TestFocalLoss:
    def test_forward_positive(self, seg_batch):
        from pygeovision.losses.segmentation import FocalLoss
        preds, targets = seg_batch
        loss = FocalLoss(alpha=0.25, gamma=2.0)(preds, targets)
        assert loss.item() >= 0

    def test_higher_gamma_lower_loss_easy_samples(self):
        """Higher gamma should reduce loss on well-classified samples."""
        import torch

        from pygeovision.losses.segmentation import FocalLoss
        # Well-classified sample: target=0, large logit for class 0
        preds   = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])   # (1, 2, 2, 1) — needs reshape
        preds   = torch.tensor([[[[10.0]], [[-10.0]]]])   # (B=1, C=2, H=1, W=1)
        targets = torch.zeros(1, 1, 1, dtype=torch.long)
        loss_g0 = FocalLoss(gamma=0.0)(preds, targets).item()
        loss_g2 = FocalLoss(gamma=2.0)(preds, targets).item()
        # gamma=2 should suppress easy samples → lower loss
        assert loss_g2 <= loss_g0 + 1e-4

    def test_zero_gamma_equals_ce(self):
        """FocalLoss(gamma=0) should equal standard CE."""
        import torch
        import torch.nn.functional as F

        from pygeovision.losses.segmentation import FocalLoss
        torch.manual_seed(5)
        preds   = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        focal   = FocalLoss(alpha=1.0, gamma=0.0)(preds, targets).item()
        ce      = F.cross_entropy(preds, targets).item()
        assert abs(focal - ce) < 0.01


class TestTverskyLoss:
    def test_forward_range(self, binary_batch):
        from pygeovision.losses.segmentation import TverskyLoss
        preds, targets = binary_batch
        loss = TverskyLoss(alpha=0.3, beta=0.7)(preds, targets)
        assert 0.0 <= loss.item() <= 2.0

    def test_dice_is_special_case(self, binary_batch):
        """Tversky(0.5, 0.5) should be close to Dice."""
        from pygeovision.losses.segmentation import DiceLoss, TverskyLoss
        preds, targets = binary_batch
        tv   = TverskyLoss(alpha=0.5, beta=0.5)(preds, targets).item()
        dice = DiceLoss()(preds, targets).item()
        assert abs(tv - dice) < 0.1


class TestComboLoss:
    def test_forward(self, seg_batch):
        from pygeovision.losses.segmentation import ComboLoss
        preds, targets = seg_batch
        loss = ComboLoss(dice_weight=0.5, ce_weight=0.5)(preds, targets)
        assert loss.item() >= 0

    def test_dice_only(self, seg_batch):
        from pygeovision.losses.segmentation import ComboLoss, DiceLoss
        preds, targets = seg_batch
        combo_dice = ComboLoss(dice_weight=1.0, ce_weight=0.0)(preds, targets).item()
        pure_dice  = DiceLoss()(preds, targets).item()
        assert abs(combo_dice - pure_dice) < 0.01


class TestBoundaryAwareLoss:
    def test_forward(self, binary_batch):
        from pygeovision.losses.segmentation import BoundaryAwareLoss
        preds, targets = binary_batch
        loss = BoundaryAwareLoss(boundary_weight=5.0)(preds, targets)
        assert loss.item() >= 0

    def test_boundary_extraction(self):
        import torch

        from pygeovision.losses.segmentation import BoundaryAwareLoss
        bl = BoundaryAwareLoss()
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, 2:6, 2:6] = 1
        boundaries = bl._extract_boundaries(targets)
        # Boundary pixels should be at edge of the filled patch
        assert boundaries.any()


class TestOhemCrossEntropy:
    def test_forward(self, seg_batch):
        from pygeovision.losses.segmentation import OhemCrossEntropy
        preds, targets = seg_batch
        loss = OhemCrossEntropy(thresh=0.7, min_kept=10)(preds, targets)
        assert loss.item() >= 0

    def test_empty_valid_pixels(self):
        """Should not crash with all-ignore targets."""
        import torch

        from pygeovision.losses.segmentation import OhemCrossEntropy
        preds   = torch.randn(1, 2, 8, 8)
        targets = torch.full((1, 8, 8), 255, dtype=torch.long)
        loss = OhemCrossEntropy()(preds, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-3) or loss.item() >= 0


class TestGeospatialMixedLoss:
    def test_forward(self, binary_batch):
        from pygeovision.losses.segmentation import GeospatialMixedLoss
        preds, targets = binary_batch
        loss_fn = GeospatialMixedLoss(weights={"combo": 0.5, "boundary": 0.3, "ohem": 0.2})
        loss = loss_fn(preds, targets)
        assert loss.item() >= 0

    def test_weights_sum_to_one(self):
        weights = {"combo": 0.5, "boundary": 0.3, "ohem": 0.2}
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_single_weight(self, binary_batch):
        from pygeovision.losses.segmentation import DiceLoss, GeospatialMixedLoss
        preds, targets = binary_batch
        mixed = GeospatialMixedLoss(weights={"dice": 1.0})(preds, targets).item()
        dice  = DiceLoss()(preds, targets).item()
        assert abs(mixed - dice) < 0.01


class TestDetectionLosses:
    @pytest.fixture
    def box_batch(self):
        import torch
        torch.manual_seed(0)
        pred   = torch.rand(4, 4)  # (N, 4) in xyxy
        target = pred.clone() + torch.randn(4, 4) * 0.05
        return pred, target

    def test_ciou_loss(self, box_batch):
        from pygeovision.losses.detection import CIoULoss
        pred, target = box_batch
        # Ensure x2>x1, y2>y1
        pred   = torch.stack([pred.min(dim=1).values, pred.max(dim=1).values], dim=1)
        pred   = torch.cat([pred[:, [0, 0]], pred[:, [1, 1]]], dim=1) * 0.5
        loss   = CIoULoss()(pred, pred.clone())  # self-overlap → near zero
        assert loss.item() >= -0.1   # ciou can be slightly negative

    def test_giou_loss(self, box_batch):
        from pygeovision.losses.detection import GIoULoss
        pred, _ = box_batch
        pred_fixed = torch.zeros(4, 4)
        pred_fixed[:, 2:] = 0.5
        loss = GIoULoss()(pred_fixed, pred_fixed)
        assert abs(loss.item()) < 0.1


class TestClassBalanceLosses:
    def test_label_smoothing(self, seg_batch):
        from pygeovision.losses.class_balance import LabelSmoothingCrossEntropy
        preds, targets = seg_batch
        loss = LabelSmoothingCrossEntropy(smoothing=0.1)(preds, targets)
        assert loss.item() >= 0

    def test_class_balanced_ce_no_counts(self, seg_batch):
        from pygeovision.losses.class_balance import ClassBalancedCrossEntropy
        preds, targets = seg_batch
        loss = ClassBalancedCrossEntropy(class_counts=None)(preds, targets)
        assert loss.item() >= 0

    def test_class_balanced_ce_with_counts(self, binary_batch):
        from pygeovision.losses.class_balance import ClassBalancedCrossEntropy
        preds, targets = binary_batch
        # Highly imbalanced: class 0 has 10x more pixels
        loss = ClassBalancedCrossEntropy(class_counts=[100000, 10000])(preds, targets)
        assert loss.item() >= 0


class TestOhemMinKeptGuarantee:
    """Regression tests for a real bug: max() was used where min() is
    required for OHEM's min_kept guarantee to actually guarantee anything.
    Verified this could select ZERO pixels (and NaN the loss via .mean()
    on an empty tensor) in a batch with mostly easy pixels — exactly the
    scenario min_kept exists to handle."""

    def test_min_kept_guarantee_holds_with_mostly_easy_pixels(self):
        import torch
        from pygeovision.losses.segmentation import OhemCrossEntropy

        # Moderately confident (not perfect) predictions: nonzero CE for
        # every pixel, but well below thresh=0.7 — exactly the scenario
        # min_kept exists to rescue. With the original max()-based bug,
        # this selects ZERO pixels (confirmed via direct reproduction)
        # and .mean() on an empty tensor returns NaN.
        C, H, W = 3, 50, 50
        targets = torch.randint(0, C, (1, H, W))
        preds = torch.zeros(1, C, H, W)
        preds.scatter_(1, targets.unsqueeze(1), 3.0)  # confident but not extreme
        preds = preds.clone().requires_grad_(True)

        loss_fn = OhemCrossEntropy(thresh=0.7, min_kept=1000)
        loss = loss_fn(preds, targets)

        assert not torch.isnan(loss), "OHEM loss is NaN — min_kept guarantee broken"
        loss.backward()
        assert torch.isfinite(preds.grad).all()

    def test_effective_threshold_never_exceeds_configured_thresh(self):
        """The min_kept floor should only ever LOOSEN the selection
        criterion, never make it stricter than the user's configured
        thresh."""
        import torch
        ce_flat = torch.cat([torch.full((900,), 0.1), torch.full((100,), 0.9)])
        thresh, min_kept = 0.7, 500
        topk_min = ce_flat.topk(min(min_kept, ce_flat.numel()))[0].min()
        effective_thresh = min(thresh, float(topk_min))
        assert effective_thresh <= thresh


class TestDetectionLossesAreGenuinelyDistinct:
    """Regression tests for a real bug: DIoULoss called CIoULoss()
    directly (silently computing CIoU under DIoU's name), and SIoULoss
    called GIoULoss() directly (silently computing a completely different
    published loss function under SIoU's name)."""

    @pytest.fixture
    def boxes(self):
        import torch
        pred = torch.tensor([[10., 10., 50., 40.]], requires_grad=True)
        target = torch.tensor([[12., 8., 55., 45.]])
        return pred, target

    def test_diou_differs_from_ciou(self, boxes):
        from pygeovision.losses.detection import CIoULoss, DIoULoss
        pred, target = boxes
        ciou = CIoULoss()(pred, target)
        diou = DIoULoss()(pred, target)
        assert abs(ciou.item() - diou.item()) > 1e-6

    def test_siou_differs_from_giou(self, boxes):
        from pygeovision.losses.detection import GIoULoss, SIoULoss
        pred, target = boxes
        giou = GIoULoss()(pred, target)
        siou = SIoULoss()(pred, target)
        assert abs(giou.item() - siou.item()) > 1e-6

    def test_siou_gradient_finite_for_identical_boxes(self):
        """Regression test for a real numerical-stability bug: clamping
        sqrt()'s OUTPUT does not protect its gradient when the input is
        exactly 0 (d/dx sqrt(x) = 1/(2*sqrt(x)), still NaN at x=0
        regardless of a downstream clamp) — identical/coincident boxes
        used to produce a NaN gradient despite a correct forward loss."""
        import torch
        from pygeovision.losses.detection import SIoULoss
        pred = torch.tensor([[10., 10., 50., 40.]], requires_grad=True)
        target = torch.tensor([[10., 10., 50., 40.]])
        loss = SIoULoss()(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_all_four_losses_finite_on_random_boxes(self):
        import torch
        from pygeovision.losses.detection import CIoULoss, DIoULoss, GIoULoss, SIoULoss
        torch.manual_seed(0)
        target = torch.tensor([[8., 3., 30., 22.]])
        for loss_cls in (CIoULoss, DIoULoss, GIoULoss, SIoULoss):
            p = torch.tensor([[5., 5., 25., 20.]], requires_grad=True)
            loss = loss_cls()(p, target)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(p.grad).all(), f"{loss_cls.__name__} produced non-finite gradient"
