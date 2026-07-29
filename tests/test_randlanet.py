"""Tests for the real RandLA-Net implementation — Hu et al. 2020.

Real per-point LiDAR segmentation tests: every test does a genuine forward
pass through Local Spatial Encoding, attentive pooling, dilated residual
blocks, and the random-sample encoder/decoder with skip connections.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


class TestRandLANetCore:
    def test_knn_point_shape(self):
        from pygeovision.models._3d.randlanet import knn_point
        torch.manual_seed(0)
        xyz = torch.randn(2, 100, 3)
        idx = knn_point(16, xyz, xyz)
        assert idx.shape == (2, 100, 16)

    def test_knn_point_clamps_k_to_available_points(self):
        """Regression test for a real bug: with a point count smaller than
        the configured k, topk(k=16, ...) on a dimension of size < 16 used
        to crash outright — must clamp instead."""
        from pygeovision.models._3d.randlanet import knn_point
        xyz = torch.randn(1, 5, 3)  # only 5 points, k=16 requested
        idx = knn_point(16, xyz, xyz)
        assert idx.shape == (1, 5, 5)  # clamped to the 5 available points

    def test_random_sample_returns_requested_count(self):
        from pygeovision.models._3d.randlanet import random_sample
        torch.manual_seed(0)
        xyz = torch.randn(2, 200, 3)
        feat = torch.randn(2, 200, 8)
        new_xyz, new_feat, idx = random_sample(xyz, feat, 50)
        assert new_xyz.shape == (2, 50, 3)
        assert new_feat.shape == (2, 50, 8)
        assert idx.shape == (2, 50)

    def test_random_sample_indices_are_distinct_per_batch(self):
        from pygeovision.models._3d.randlanet import random_sample
        torch.manual_seed(0)
        xyz = torch.randn(1, 500, 3)
        _, _, idx = random_sample(xyz, xyz, 100)
        assert len(torch.unique(idx[0])) == 100  # sampling without replacement


class TestRandLANetSegmentation:
    """Real end-to-end forward-pass tests for the segmentation network."""

    def test_output_shape(self):
        from pygeovision.models._3d.randlanet import build_randlanet
        torch.manual_seed(0)
        model = build_randlanet(num_classes=5, in_channels=4, k=16)
        model.eval()
        xyz = torch.randn(2, 4, 2048)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (2, 5, 2048)

    def test_xyz_only_no_extra_features(self):
        from pygeovision.models._3d.randlanet import build_randlanet
        model = build_randlanet(num_classes=3, in_channels=3, k=16)
        model.eval()
        xyz = torch.randn(1, 3, 512)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 512)

    def test_extra_features_channel_count(self):
        from pygeovision.models._3d.randlanet import build_randlanet
        model = build_randlanet(num_classes=4, in_channels=6, k=16)
        model.eval()
        xyz = torch.randn(1, 6, 512)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 4, 512)

    def test_small_point_cloud_does_not_crash(self):
        """Regression test for the real knn/LocSE clamping bugs: a point
        count that drops below k after repeated 4x decimation used to
        crash (topk overflow, then a LocSE shape mismatch after the first
        fix). Must complete cleanly end-to-end now."""
        from pygeovision.models._3d.randlanet import build_randlanet
        model = build_randlanet(num_classes=3, in_channels=3, k=16)
        model.eval()
        xyz = torch.randn(1, 3, 256)  # 256 -> 64 -> 16 -> 4 -> 1 across 4 layers
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 256)

    def test_gradient_flows_through_all_parameters(self):
        from pygeovision.models._3d.randlanet import build_randlanet
        torch.manual_seed(1)
        model = build_randlanet(num_classes=5, in_channels=4, k=16)
        xyz = torch.randn(1, 4, 1024)
        out = model(xyz)
        loss = (out ** 2).mean()
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.grad is None or p.grad.abs().sum() == 0]
        assert missing == [], f"parameters with no gradient: {missing}"

    def test_different_point_counts_all_work(self):
        from pygeovision.models._3d.randlanet import build_randlanet
        model = build_randlanet(num_classes=3, in_channels=3, k=16)
        model.eval()
        for n in (256, 1000, 4096):
            xyz = torch.randn(1, 3, n)
            with torch.no_grad():
                out = model(xyz)
            assert out.shape == (1, 3, n)

    def test_registry_dispatch(self):
        from pygeovision.models.registry import get_model
        model = get_model("randlanet", num_classes=5, in_channels=4, pretrained=False)
        model.eval()
        xyz = torch.randn(1, 4, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 5, 1024)
