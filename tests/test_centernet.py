"""CenterNet: target encoding, decoding and a small end-to-end fit.

Targets and decoding are checked against hand-computed values; the training test
fits a handful of synthetic square "objects" and requires the detections to overlap
the ground truth, so a wrong stride, label shift or box convention fails it.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torchvision", reason="torchvision not installed")

from pygeovision.models.detection.centernet import (  # noqa: E402
    CenterNet,
    build_centernet,
    gaussian_radius,
)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TestGaussianRadius:
    @pytest.mark.parametrize("h,w", [(20.0, 20.0), (50.0, 30.0), (100.0, 100.0)])
    def test_matches_independent_quadratic_solution(self, h, w):
        o = 0.7
        # the three corner-shift cases of CornerNet, solved explicitly
        r1 = (( h + w) + math.sqrt((h + w) ** 2 - 4 * (w * h * (1 - o) / (1 + o)))) / 2
        r2 = (2 * (h + w) + math.sqrt((2 * (h + w)) ** 2 - 4 * 4 * ((1 - o) * w * h))) / 2
        r3 = (-2 * o * (h + w) + math.sqrt((2 * o * (h + w)) ** 2 - 4 * 4 * o * ((o - 1) * w * h))) / 2
        assert gaussian_radius(h, w, o) == pytest.approx(min(r1, r2, r3))

    def test_known_value_from_the_reference_formula(self):
        # 40x40 box, min_overlap 0.7: r1=76.3, r2=..., r3=10.93 -> the reference returns the minimum.
        assert gaussian_radius(40, 40, 0.7) == pytest.approx(10.9328, abs=1e-3)

    def test_larger_objects_get_larger_radius(self):
        assert gaussian_radius(80, 80) > gaussian_radius(20, 20)


class TestTargets:
    def test_peak_size_and_offset_at_expected_cell(self):
        m = CenterNet(num_classes=2, in_channels=3)
        box = torch.tensor([[20.0, 36.0, 52.0, 60.0]])   # w=32 h=24, centre (36, 48)
        hm, wh, off, mask = m._targets([{"boxes": box, "labels": torch.tensor([2])}], (32, 32),
                                       torch.device("cpu"))
        cx, cy = 36 / 4, 48 / 4                          # 9.0, 12.0 -> cell (9, 12)
        assert hm[0, 1, 12, 9].item() == pytest.approx(1.0)
        assert hm[0, 0].max().item() == 0.0              # other class untouched
        assert hm[0, 1].argmax().item() == 12 * 32 + 9
        assert wh[0, :, 12, 9].tolist() == pytest.approx([8.0, 6.0])      # (w, h) / stride
        assert off[0, :, 12, 9].tolist() == pytest.approx([cx - 9, cy - 12])
        assert mask.sum().item() == 1

    def test_fractional_centre_gives_offset(self):
        m = CenterNet(num_classes=1, in_channels=3)
        box = torch.tensor([[17.0, 17.0, 33.0, 33.0]])   # centre (25, 25) -> 6.25
        _, _, off, _ = m._targets([{"boxes": box, "labels": torch.tensor([1])}], (16, 16),
                                  torch.device("cpu"))
        assert off[0, :, 6, 6].tolist() == pytest.approx([0.25, 0.25])

    def test_label_zero_and_out_of_range_rejected(self):
        m = CenterNet(num_classes=2, in_channels=3)
        box = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
        for bad in (0, 3):
            with pytest.raises(ValueError):
                m._targets([{"boxes": box, "labels": torch.tensor([bad])}], (8, 8), torch.device("cpu"))


class TestDecode:
    def test_decodes_a_planted_peak_to_the_expected_box(self):
        m = CenterNet(num_classes=3, in_channels=3, score_thresh=0.5).eval()
        hm = torch.full((1, 3, 16, 16), -10.0)
        wh = torch.zeros(1, 2, 16, 16)
        off = torch.zeros(1, 2, 16, 16)
        hm[0, 2, 5, 7] = 8.0                      # class index 2 -> label 3
        wh[0, :, 5, 7] = torch.tensor([6.0, 4.0])
        off[0, :, 5, 7] = torch.tensor([0.5, 0.25])
        out = m._decode((hm, wh, off), [(64, 64)])[0]
        assert out["labels"].tolist() == [3]
        cx, cy = (7 + 0.5) * 4, (5 + 0.25) * 4    # 30, 21
        assert out["boxes"][0].tolist() == pytest.approx([cx - 12, cy - 8, cx + 12, cy + 8])
        assert out["scores"][0].item() > 0.99

    def test_boxes_clipped_to_the_image(self):
        m = CenterNet(num_classes=1, in_channels=3, score_thresh=0.5).eval()
        hm = torch.full((1, 1, 8, 8), -10.0)
        wh = torch.zeros(1, 2, 8, 8)
        hm[0, 0, 0, 0] = 8.0
        wh[0, :, 0, 0] = 10.0
        box = m._decode((hm, wh, torch.zeros(1, 2, 8, 8)), [(32, 32)])[0]["boxes"][0]
        assert box[0].item() == 0.0 and box[1].item() == 0.0

    def test_only_local_maxima_survive(self):
        m = CenterNet(num_classes=1, in_channels=3, score_thresh=0.1).eval()
        hm = torch.full((1, 1, 8, 8), -10.0)
        hm[0, 0, 4, 4] = 6.0
        hm[0, 0, 4, 5] = 5.0                      # neighbour of a stronger peak: suppressed
        wh = torch.ones(1, 2, 8, 8)
        out = m._decode((hm, wh, torch.zeros(1, 2, 8, 8)), [(32, 32)])[0]
        assert len(out["boxes"]) == 1


class TestForwardContract:
    @pytest.mark.parametrize("ch", [3, 4, 13])
    def test_any_band_count(self, ch):
        m = build_centernet(num_classes=2, in_channels=ch, backbone="resnet18").eval()
        out = m([torch.rand(ch, 96, 96)])
        assert set(out[0]) == {"boxes", "scores", "labels"}

    def test_mixed_size_list_is_padded_and_clipped(self):
        torch.manual_seed(0)
        m = build_centernet(num_classes=1, in_channels=3, backbone="resnet18", score_thresh=0.0).eval()
        # A random-init size head predicts ~0-sized boxes, which are (correctly) dropped as
        # degenerate -- the old version of this test then crashed on max() of an empty tensor
        # about 60% of the time.  Make the boxes large so the clipping is what is tested.
        with torch.no_grad():
            m.wh_head[-1].bias.fill_(40.0)
        out = m([torch.rand(3, 64, 96), torch.rand(3, 80, 50)])
        assert len(out) == 2
        assert len(out[0]["boxes"]) > 0 and len(out[1]["boxes"]) > 0
        assert out[1]["boxes"][:, 2].max() <= 50 and out[1]["boxes"][:, 3].max() <= 80
        assert out[0]["boxes"][:, 2].max() <= 96 and out[0]["boxes"][:, 3].max() <= 64
        # and the boxes really were clipped, not merely small
        assert out[1]["boxes"][:, 2].max() == 50

    def test_training_requires_targets(self):
        m = build_centernet(num_classes=1, in_channels=3, backbone="resnet18").train()
        with pytest.raises(ValueError):
            m([torch.rand(3, 64, 64)])

    def test_wrong_channel_count_rejected(self):
        m = build_centernet(num_classes=1, in_channels=4, backbone="resnet18").eval()
        with pytest.raises(ValueError, match="C=4"):
            m([torch.rand(3, 64, 64)])

    def test_unknown_backbone_rejected(self):
        with pytest.raises(ValueError):
            CenterNet(1, 3, backbone="vgg16")

    def test_registered_in_the_model_registry(self):
        from pygeovision.models.registry import get_model
        m = get_model("centernet-r50", num_classes=2, in_channels=4, pretrained=False)
        assert isinstance(m, CenterNet)


def _scene(rng, size, boxes, labels, ch=4):
    img = rng.normal(0.15, 0.03, (ch, size, size)).astype(np.float32)
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        img[:, y1:y2, x1:x2] += 0.6 if lab == 1 else 0.0
        if lab == 2:
            img[0, y1:y2, x1:x2] += 0.7
            img[1:, y1:y2, x1:x2] -= 0.05
    return torch.from_numpy(img).clamp(0, 1)


def test_fits_synthetic_objects_and_detects_them():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    layouts = [
        ([(16, 20, 48, 52), (80, 70, 112, 110)], [1, 2]),
        ([(60, 10, 100, 40)], [1]),
        ([(10, 70, 40, 120), (70, 20, 110, 50)], [2, 1]),
        ([(30, 30, 70, 62)], [2]),
    ]
    images = [_scene(rng, 128, b, l) for b, l in layouts]
    targets = [{"boxes": torch.tensor(b, dtype=torch.float32), "labels": torch.tensor(l)} for b, l in layouts]
    model = build_centernet(num_classes=2, in_channels=4, backbone="resnet18", score_thresh=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    first = None
    for step in range(180):
        losses = model(images, targets)
        loss = sum(losses.values())
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.35 * first, "training loss did not fall enough"

    model.eval()
    with torch.no_grad():
        dets = model(images)
    found = total = 0
    for det, tgt in zip(dets, targets):
        for gt_box, gt_lab in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()):
            total += 1
            if any(l == gt_lab and _iou(b, gt_box) > 0.5
                   for b, l in zip(det["boxes"].tolist(), det["labels"].tolist())):
                found += 1
    assert found == total, f"only {found}/{total} ground-truth objects were detected"
