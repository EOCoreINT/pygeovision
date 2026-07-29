"""
PointNet++ — Deep Hierarchical Feature Learning on Point Sets.

Faithful port of Qi, Yi, Su, Guibas (2017), "PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space" (NeurIPS 2017), official
code: https://github.com/charlesq34/pointnet2 (TensorFlow) /
https://github.com/yanx27/Pointnet_Pointnet2_pytorch (PyTorch reference).

Implements the SEGMENTATION variant (per-point class prediction) — this is
the geospatial LiDAR use case (classify each point as ground / vegetation /
building / water, etc.), not whole-cloud classification.

Architecture:
  1. Set Abstraction (SA) layers hierarchically downsample the point cloud:
     - Sampling: Farthest Point Sampling (FPS) selects well-spread centroids
     - Grouping: ball query finds neighbors within a radius of each centroid
     - PointNet: a shared MLP + max-pool extracts a local feature per group
     SSG (Single-Scale Grouping) uses one radius/neighbor-count per layer;
     MSG (Multi-Scale Grouping) uses several in parallel and concatenates
     the results, which is more robust to non-uniform point density
     (very common in real LiDAR — near-nadir vs. edge-of-swath density
     varies a lot).
  2. Feature Propagation (FP) layers upsample back to the original point
     count via inverse-distance-weighted interpolation from the 3 nearest
     neighbors in the coarser layer, concatenated with skip-connected
     features from the corresponding SA layer (U-Net-style).
  3. A final per-point MLP produces class logits.

This is NOT a registry stub — every module below is a real, working
nn.Module with verified forward-pass shapes (see the test suite).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        raise ImportError("torch required for PointNet++: pip install torch") from None


def square_distance(src, dst):
    """Pairwise squared Euclidean distance between two point sets.
    src: (B, N, C), dst: (B, M, C) -> (B, N, M)."""
    torch, nn, F = _require_torch()
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Gather points/features by index.
    points: (B, N, C), idx: (B, S) or (B, S, K) -> (B, S, C) or (B, S, K, C)."""
    torch, nn, F = _require_torch()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling.
    xyz: (B, N, 3) -> centroid indices (B, npoint)."""
    torch, nn, F = _require_torch()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """For each centroid in new_xyz, find up to nsample points in xyz within
    `radius`. xyz: (B, N, 3), new_xyz: (B, S, 3) -> group indices (B, S, nsample')."""
    torch, nn, F = _require_torch()
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    nsample = min(nsample, N)  # can't return more neighbors than points exist
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, group_idx.shape[-1]])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """FPS + ball-query grouping + local (centroid-relative) coordinates."""
    torch, nn, F = _require_torch()
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """Group ALL points into a single region (used for the final SA layer)."""
    torch, nn, F = _require_torch()
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def _build_set_abstraction():
    torch, nn, F = _require_torch()

    class PointNetSetAbstraction(nn.Module):
        """Single-Scale Grouping (SSG) set abstraction layer."""
        def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
            super().__init__()
            self.npoint = npoint
            self.radius = radius
            self.nsample = nsample
            self.group_all = group_all
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

        def forward(self, xyz, points):
            """xyz: (B, C, N) coordinates, points: (B, D, N) features or None.
            Returns new_xyz (B, C, npoint), new_points (B, D', npoint)."""
            xyz = xyz.permute(0, 2, 1)
            if points is not None:
                points = points.permute(0, 2, 1)

            if self.group_all:
                new_xyz, new_points = sample_and_group_all(xyz, points)
            else:
                new_xyz, new_points = sample_and_group(
                    self.npoint, self.radius, self.nsample, xyz, points,
                )
            # new_points: (B, npoint, nsample, C+D) -> (B, C+D, nsample, npoint)
            new_points = new_points.permute(0, 3, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                new_points = F.relu(bn(conv(new_points)))
            new_points = torch.max(new_points, 2)[0]  # local max-pool over nsample
            new_xyz = new_xyz.permute(0, 2, 1)
            return new_xyz, new_points

    class PointNetSetAbstractionMsg(nn.Module):
        """Multi-Scale Grouping (MSG) set abstraction layer — runs several
        radius/nsample/mlp combinations in parallel and concatenates the
        results. More robust to the non-uniform point density typical of
        real airborne/terrestrial LiDAR scans."""
        def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
            super().__init__()
            self.npoint = npoint
            self.radius_list = radius_list
            self.nsample_list = nsample_list
            self.conv_blocks = nn.ModuleList()
            self.bn_blocks = nn.ModuleList()
            for mlp in mlp_list:
                convs = nn.ModuleList()
                bns = nn.ModuleList()
                last_channel = in_channel + 3
                for out_channel in mlp:
                    convs.append(nn.Conv2d(last_channel, out_channel, 1))
                    bns.append(nn.BatchNorm2d(out_channel))
                    last_channel = out_channel
                self.conv_blocks.append(convs)
                self.bn_blocks.append(bns)

        def forward(self, xyz, points):
            xyz = xyz.permute(0, 2, 1)
            if points is not None:
                points = points.permute(0, 2, 1)

            B, N, C = xyz.shape
            S = self.npoint
            new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

            new_points_list = []
            for i, radius in enumerate(self.radius_list):
                K = self.nsample_list[i]
                group_idx = query_ball_point(radius, K, xyz, new_xyz)
                grouped_xyz = index_points(xyz, group_idx)
                grouped_xyz -= new_xyz.view(B, S, 1, C)
                if points is not None:
                    grouped_points = index_points(points, group_idx)
                    grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                else:
                    grouped_points = grouped_xyz

                grouped_points = grouped_points.permute(0, 3, 2, 1)
                for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                    grouped_points = F.relu(bn(conv(grouped_points)))
                new_points = torch.max(grouped_points, 2)[0]
                new_points_list.append(new_points)

            new_xyz = new_xyz.permute(0, 2, 1)
            new_points_concat = torch.cat(new_points_list, dim=1)
            return new_xyz, new_points_concat

    class PointNetFeaturePropagation(nn.Module):
        """Upsamples features from a coarser (fewer-point) layer back to a
        finer layer via inverse-distance-weighted interpolation from the 3
        nearest coarse points, concatenated with the finer layer's own
        (skip-connected) features."""
        def __init__(self, in_channel, mlp):
            super().__init__()
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel

        def forward(self, xyz1, xyz2, points1, points2):
            """xyz1: (B,C,N) finer, xyz2: (B,C,S) coarser (S<=N).
            points1: (B,D1,N) finer features (or None), points2: (B,D2,S)
            coarser features. Returns (B, D', N)."""
            xyz1 = xyz1.permute(0, 2, 1)
            xyz2 = xyz2.permute(0, 2, 1)
            points2 = points2.permute(0, 2, 1)
            B, N, C = xyz1.shape
            _, S, _ = xyz2.shape

            if S == 1:
                interpolated_points = points2.repeat(1, N, 1)
            else:
                dists = square_distance(xyz1, xyz2)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]
                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm
                interpolated_points = torch.sum(
                    index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2,
                )

            if points1 is not None:
                points1 = points1.permute(0, 2, 1)
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = interpolated_points

            new_points = new_points.permute(0, 2, 1)
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                new_points = F.relu(bn(conv(new_points)))
            return new_points

    return PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation


class PointNet2SegModel:
    """Real PointNet++ segmentation network builder — per-point class logits.

    Example::

        model = PointNet2SegModel(num_classes=5, extra_feature_channels=1,
                                   msg=False)  # xyz + 1 extra feature (e.g. intensity)
        logits = model(xyz)  # xyz: (B, 3+extra, N) -> (B, num_classes, N)
    """

    def __new__(cls, num_classes: int = 5, extra_feature_channels: int = 0,
                msg: bool = False):
        torch, nn, F = _require_torch()
        SA, SAMsg, FP = _build_set_abstraction()

        class PointNet2Seg(nn.Module):
            def __init__(self):
                super().__init__()
                extra = extra_feature_channels
                if msg:
                    self.sa1 = SAMsg(1024, [0.05, 0.1], [16, 32], extra,
                                      [[16, 16, 32], [32, 32, 64]])
                    self.sa2 = SAMsg(256, [0.1, 0.2], [16, 32], 32 + 64,
                                      [[64, 64, 128], [64, 96, 128]])
                    sa2_out = 128 + 128
                else:
                    self.sa1 = SA(1024, 0.1, 32, extra + 3, [32, 32, 64], False)
                    self.sa2 = SA(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
                    sa2_out = 128

                self.sa3 = SA(None, None, None, sa2_out + 3, [128, 256, 1024], True)

                sa1_out = 64 if not msg else 32 + 64
                self.fp3 = FP(1024 + sa2_out, [256, 256])
                self.fp2 = FP(256 + sa1_out, [256, 128])
                self.fp1 = FP(128 + extra, [128, 128, 128])

                self.conv1 = nn.Conv1d(128, 128, 1)
                self.bn1 = nn.BatchNorm1d(128)
                self.drop1 = nn.Dropout(0.5)
                self.conv2 = nn.Conv1d(128, num_classes, 1)

            def forward(self, xyz):
                # xyz: (B, 3+extra, N)
                l0_xyz = xyz[:, :3, :]
                # Only the extra per-point features (beyond xyz) go in as
                # "points" — sample_and_group() re-adds local xyz coordinates
                # itself, so passing the full xyz+features tensor here would
                # double-count the xyz channels.
                l0_points = xyz[:, 3:, :] if extra_feature_channels > 0 else None

                l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

                l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
                l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
                l0_points = self.fp1(
                    l0_xyz, l1_xyz,
                    l0_points if extra_feature_channels > 0 else None,
                    l1_points,
                )

                x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
                x = self.conv2(x)
                return x  # (B, num_classes, N)

        return PointNet2Seg()


def build_pointnet2(num_classes: int = 5, in_channels: int = 3, msg: bool = False, **kwargs) -> Any:
    """Factory used by the model registry — `get_model('pointnet2-ssg', ...)`
    / `get_model('pointnet2-msg', ...)`. `in_channels` = 3 (xyz only) +
    however many extra per-point features (intensity, RGB, return number,
    etc.) — e.g. in_channels=4 for xyz+intensity."""
    extra = max(in_channels - 3, 0)
    return PointNet2SegModel(num_classes=num_classes, extra_feature_channels=extra, msg=msg)