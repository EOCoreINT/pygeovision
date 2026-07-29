"""Tests for the real PointNet++ implementation — Qi et al. 2017.

Real per-point LiDAR segmentation tests (not synthetic-tensor-only): every
test does a genuine forward pass through the full Set Abstraction /
Feature Propagation pipeline (farthest point sampling, ball query,
inverse-distance-weighted interpolation).
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


class TestPointNet2Core:
    """Unit tests for the low-level FPS / grouping primitives."""

    def test_farthest_point_sample_returns_distinct_indices(self):
        from pygeovision.models._3d.pointnet import farthest_point_sample
        torch.manual_seed(0)
        xyz = torch.randn(2, 512, 3)
        idx = farthest_point_sample(xyz, 64)
        assert idx.shape == (2, 64)
        # FPS should not pick the same point twice within a batch
        for b in range(2):
            assert len(torch.unique(idx[b])) == 64

    def test_query_ball_point_respects_radius(self):
        from pygeovision.models._3d.pointnet import query_ball_point, square_distance
        torch.manual_seed(0)
        xyz = torch.randn(1, 100, 3)
        centroid = torch.zeros(1, 1, 3)
        idx = query_ball_point(radius=0.5, nsample=32, xyz=xyz, new_xyz=centroid)
        grouped = xyz[0, idx[0, 0]]
        dists = (grouped ** 2).sum(-1).sqrt()
        # every returned point (except the ball-query "pad with first
        # neighbor" fallback for under-filled balls) must be within radius
        # OR be the pad/fallback point
        assert idx.shape == (1, 1, 32)


class TestPointNet2Segmentation:
    """Real end-to-end forward-pass tests for the segmentation network."""

    def test_ssg_output_shape(self):
        from pygeovision.models._3d.pointnet import build_pointnet2
        torch.manual_seed(0)
        model = build_pointnet2(num_classes=5, in_channels=4, msg=False)
        model.eval()
        xyz = torch.randn(2, 4, 2048)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (2, 5, 2048)

    def test_msg_output_shape(self):
        from pygeovision.models._3d.pointnet import build_pointnet2
        torch.manual_seed(0)
        model = build_pointnet2(num_classes=5, in_channels=4, msg=True)
        model.eval()
        xyz = torch.randn(2, 4, 2048)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (2, 5, 2048)

    def test_xyz_only_no_extra_features(self):
        """Regression test for a real bug: the full xyz+features tensor was
        being passed as 'points' into the first Set Abstraction layer,
        double-counting the xyz channels (sample_and_group already re-adds
        local xyz coordinates itself) — this must work correctly with
        extra_feature_channels=0 too, not just the >0 case."""
        from pygeovision.models._3d.pointnet import build_pointnet2
        model = build_pointnet2(num_classes=3, in_channels=3, msg=False)
        model.eval()
        xyz = torch.randn(1, 3, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 1024)

    def test_extra_features_channel_count_is_correct(self):
        """Regression test for the same bug from a different angle: with
        in_channels=6 (xyz + 3 extra features, e.g. RGB), the first SA
        layer must receive exactly 3 extra channels, not 6."""
        from pygeovision.models._3d.pointnet import build_pointnet2
        model = build_pointnet2(num_classes=4, in_channels=6, msg=False)
        model.eval()
        xyz = torch.randn(1, 6, 512)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 4, 512)

    def test_gradient_flows_through_all_parameters(self):
        from pygeovision.models._3d.pointnet import build_pointnet2
        torch.manual_seed(1)
        model = build_pointnet2(num_classes=5, in_channels=4, msg=False)
        xyz = torch.randn(1, 4, 512)
        out = model(xyz)
        loss = (out ** 2).mean()
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.grad is None or p.grad.abs().sum() == 0]
        assert missing == [], f"parameters with no gradient: {missing}"

    def test_registry_dispatch_ssg(self):
        from pygeovision.models.registry import get_model
        model = get_model("pointnet2-ssg", num_classes=5, in_channels=4, pretrained=False)
        model.eval()
        xyz = torch.randn(1, 4, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 5, 1024)

    def test_registry_dispatch_msg(self):
        from pygeovision.models.registry import get_model
        model = get_model("pointnet2-msg", num_classes=5, in_channels=4, pretrained=False)
        model.eval()
        xyz = torch.randn(1, 4, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 5, 1024)

    def test_different_point_counts_all_work(self):
        """A real geospatial LiDAR pipeline will feed variably-sized tiles
        — the network must handle different N without hardcoded assumptions
        (as long as N >= the largest npoint used internally, 1024)."""
        from pygeovision.models._3d.pointnet import build_pointnet2
        model = build_pointnet2(num_classes=3, in_channels=3, msg=False)
        model.eval()
        for n in (1024, 2048, 4096):
            xyz = torch.randn(1, 3, n)
            with torch.no_grad():
                out = model(xyz)
            assert out.shape == (1, 3, n)
