"""Tests for the real KPConv (rigid kernel) implementation — Thomas et al. 2019.

Two real bugs found and fixed during implementation:
  1. Strided-block shortcuts and decoder upsampling used query_ball_point
     with a huge radius and nsample=1 to approximate "nearest neighbor" —
     but query_ball_point sorts candidates by index, not distance, so this
     always returned an arbitrary (lowest-index) point regardless of query
     location, not the genuine nearest neighbor. Needed a real knn_point.
  2. query_ball_point's group_first.repeat(nsample) used the requested
     nsample even when the actual post-slice group_idx had fewer columns
     (support point count < nsample) — a shape mismatch. This bug was
     ALSO latent in the PointNet++ implementation (same function, copied),
     just never triggered by tests that always used point counts > nsample.
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


class TestKPConvCore:
    def test_generate_kernel_points_shape(self):
        from pygeovision.models._3d.kpconv import generate_kernel_points
        pts = generate_kernel_points(radius=0.1, num_kpoints=15)
        assert pts.shape == (15, 3)

    def test_kernel_points_within_radius(self):
        from pygeovision.models._3d.kpconv import generate_kernel_points
        pts = generate_kernel_points(radius=0.1, num_kpoints=15)
        dists = torch.norm(pts, dim=-1)
        assert (dists <= 0.1 + 1e-6).all()

    def test_knn_point_is_genuinely_distance_based(self):
        """Regression test for the real bug: the old shortcut/upsampling
        code used query_ball_point(huge_radius, nsample=1, ...), which
        sorts by INDEX not distance, so every query point got the same
        arbitrary (lowest-index) match regardless of its actual location."""
        from pygeovision.models._3d.kpconv import knn_point
        support = torch.tensor([[[0, 0, 0], [10, 10, 10], [5, 5, 5], [1, 1, 1]]],
                                dtype=torch.float32)
        query = torch.tensor([[[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]]], dtype=torch.float32)
        idx = knn_point(1, support, query)
        # Query near origin must match support point 0; query near [10,10,10]
        # must match support point 1 — genuinely different, distance-based.
        assert idx[0, 0, 0].item() == 0
        assert idx[0, 1, 0].item() == 1

    def test_query_ball_point_handles_fewer_points_than_nsample(self):
        """Regression test: with only 5 support points and nsample=32
        requested, this used to crash with a shape mismatch in
        group_first.repeat(nsample) — must clamp gracefully instead."""
        from pygeovision.models._3d.kpconv import query_ball_point
        xyz = torch.randn(1, 5, 3)
        idx, mask = query_ball_point(radius=10.0, nsample=32, xyz=xyz, new_xyz=xyz)
        assert idx.shape == (1, 5, 5)  # clamped to the 5 available points


class TestKPConvLayer:
    def test_single_layer_forward_shape(self):
        from pygeovision.models._3d.kpconv import _build_kpconv_layer
        KPConvLayer = _build_kpconv_layer()
        torch.manual_seed(0)
        layer = KPConvLayer(in_channels=8, out_channels=16, radius=0.2)
        xyz = torch.randn(2, 100, 3)
        feat = torch.randn(2, 8, 100)
        out = layer(xyz, xyz, feat)
        assert out.shape == (2, 16, 100)

    def test_strided_layer_downsamples_correctly(self):
        """Query points can be a smaller subset than support points
        (strided convolution) — output must match the query count."""
        from pygeovision.models._3d.kpconv import _build_kpconv_layer
        KPConvLayer = _build_kpconv_layer()
        layer = KPConvLayer(in_channels=8, out_channels=16, radius=0.5)
        support_xyz = torch.randn(1, 200, 3)
        query_xyz = support_xyz[:, :50, :]  # first 50 points as query
        feat = torch.randn(1, 8, 200)
        out = layer(query_xyz, support_xyz, feat)
        assert out.shape == (1, 16, 50)


class TestKPConvSegmentation:
    """Real end-to-end forward-pass tests for the segmentation network."""

    def test_output_shape(self):
        from pygeovision.models._3d.kpconv import build_kpconv
        torch.manual_seed(0)
        model = build_kpconv(num_classes=5, in_channels=4, base_radius=0.1, num_layers=3)
        model.eval()
        xyz = torch.randn(2, 4, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (2, 5, 1024)

    def test_xyz_only_no_extra_features(self):
        from pygeovision.models._3d.kpconv import build_kpconv
        model = build_kpconv(num_classes=3, in_channels=3, base_radius=0.1, num_layers=3)
        model.eval()
        xyz = torch.randn(1, 3, 512)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 512)

    def test_small_point_cloud_does_not_crash(self):
        """Regression test for the query_ball_point shape-mismatch bug —
        small inputs used to crash partway through the encoder."""
        from pygeovision.models._3d.kpconv import build_kpconv
        model = build_kpconv(num_classes=3, in_channels=3, base_radius=0.1, num_layers=3)
        model.eval()
        xyz = torch.randn(1, 3, 128)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 128)

    def test_gradient_flows_through_all_parameters(self):
        from pygeovision.models._3d.kpconv import build_kpconv
        torch.manual_seed(1)
        model = build_kpconv(num_classes=5, in_channels=4, base_radius=0.1, num_layers=3)
        xyz = torch.randn(1, 4, 512)
        out = model(xyz)
        loss = (out ** 2).mean()
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.grad is None or p.grad.abs().sum() == 0]
        assert missing == [], f"parameters with no gradient: {missing}"

    def test_different_point_counts_all_work(self):
        from pygeovision.models._3d.kpconv import build_kpconv
        model = build_kpconv(num_classes=3, in_channels=3, base_radius=0.1, num_layers=3)
        model.eval()
        for n in (128, 256, 1024, 2048):
            xyz = torch.randn(1, 3, n)
            with torch.no_grad():
                out = model(xyz)
            assert out.shape == (1, 3, n)

    def test_registry_dispatch(self):
        from pygeovision.models.registry import get_model
        model = get_model("kpconv", num_classes=5, in_channels=4, pretrained=False)
        model.eval()
        xyz = torch.randn(1, 4, 1024)
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 5, 1024)


class TestPointNet2QueryBallPointRegression:
    """The same query_ball_point bug was latent in pointnet.py (identical
    function, never triggered because prior tests always used point
    counts larger than nsample) — verify it's fixed there too."""

    def test_pointnet2_handles_small_point_cloud(self):
        from pygeovision.models._3d.pointnet import build_pointnet2
        model = build_pointnet2(num_classes=3, in_channels=3, msg=False)
        model.eval()
        xyz = torch.randn(1, 3, 64)  # small enough to have tripped the old bug
        with torch.no_grad():
            out = model(xyz)
        assert out.shape == (1, 3, 64)

    def test_pointnet2_query_ball_point_clamps_nsample(self):
        from pygeovision.models._3d.pointnet import query_ball_point
        xyz = torch.randn(1, 5, 3)
        idx = query_ball_point(radius=10.0, nsample=32, xyz=xyz, new_xyz=xyz)
        assert idx.shape == (1, 5, 5)
