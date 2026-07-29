"""
RandLA-Net — Efficient Semantic Segmentation of Large-Scale Point Clouds.

Faithful port of Hu et al. (2020), "RandLA-Net: Efficient Semantic
Segmentation of Large-Scale Point Clouds" (CVPR 2020 Oral), official code:
https://github.com/QingyongHu/RandLA-Net

Designed specifically for the scale of real airborne/terrestrial LiDAR —
the paper reports processing 1M points in a single pass, up to 200x faster
than prior point-cloud networks, because it uses random sampling instead
of farthest point sampling (which is the main cost driver in PointNet++).
Random sampling alone can discard important points by chance, so RandLA-Net
compensates with a Local Feature Aggregation (LFA) module:

  1. Local Spatial Encoding (LocSE): for each point, finds its K nearest
     neighbors and explicitly encodes each neighbor's relative position
     (center xyz, neighbor xyz, relative offset, Euclidean distance),
     concatenated with the neighbor's own features — so every point stays
     aware of local geometry even after aggressive downsampling.
  2. Attentive Pooling: instead of max-pooling over the K neighbors (as
     PointNet++ does), a learned attention mechanism scores each neighbor
     and computes a weighted sum — avoiding the information loss of taking
     just the single strongest activation per channel.
  3. Dilated Residual Block: two LocSE + Attentive Pooling units stacked
     with a skip connection (ResNet-style), which cheaply expands each
     point's effective receptive field to its two-hop neighborhood.

Encoder: 4 stages of [Dilated Residual Block -> Random Sampling downsample].
Decoder: nearest-neighbor upsampling + skip connections back to each
encoder stage, symmetric U-Net style, ending in a per-point classifier.

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
        raise ImportError("torch required for RandLA-Net: pip install torch") from None


def square_distance(src, dst):
    """Pairwise squared Euclidean distance. src:(B,N,C) dst:(B,M,C) -> (B,N,M)."""
    torch, nn, F = _require_torch()
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist.clamp(min=0)


def index_points(points, idx):
    """Gather points/features by index. points:(B,N,C) idx:(B,S) or (B,S,K)."""
    torch, nn, F = _require_torch()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def knn_point(k, xyz, new_xyz):
    """k nearest neighbors of each point in new_xyz, among xyz.
    xyz:(B,N,3) new_xyz:(B,S,3) -> indices (B,S,k').

    k is clamped to min(k, N) — with repeated downsampling (RandLA-Net's
    encoder decimates by 4x per layer), the point count can legitimately
    drop below the configured k before the encoder finishes; every layer
    here handles a runtime k' <= k without needing fixed-k assumptions
    baked into its channel dimensions.
    """
    torch, nn, F = _require_torch()
    k = min(k, xyz.shape[1])
    dists = square_distance(new_xyz, xyz)
    idx = dists.topk(k, dim=-1, largest=False)[1]
    return idx


def random_sample(xyz, features, n_sample):
    """Randomly select n_sample points (uniform, without replacement) —
    the key efficiency trick of RandLA-Net vs. FPS-based methods.
    xyz:(B,N,3) features:(B,N,C) -> new_xyz, new_features, sampled indices."""
    torch, nn, F = _require_torch()
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.stack([torch.randperm(N, device=device)[:n_sample] for _ in range(B)])
    new_xyz = index_points(xyz, idx)
    new_features = index_points(features, idx) if features is not None else None
    return new_xyz, new_features, idx


def _build_lfa_modules():
    torch, nn, F = _require_torch()

    class LocSE(nn.Module):
        """Local Spatial Encoding — augments each of the K neighbor
        features with an explicit encoding of its relative position."""
        def __init__(self, feature_dim, k):
            super().__init__()
            self.k = k
            self.mlp = nn.Sequential(
                nn.Conv2d(10, feature_dim, 1), nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
            )

        def forward(self, xyz, features):
            """xyz:(B,N,3) features:(B,C,N) -> augmented (B, 2C, N, k')."""
            B, N, _ = xyz.shape
            idx = knn_point(self.k, xyz, xyz)                       # (B, N, k') — k' = min(self.k, N)
            k_actual = idx.shape[-1]
            neighbor_xyz = index_points(xyz, idx)                    # (B, N, k', 3)
            center_xyz = xyz.unsqueeze(2).repeat(1, 1, k_actual, 1)   # (B, N, k', 3)
            relative = center_xyz - neighbor_xyz
            dist = torch.norm(relative, dim=-1, keepdim=True)
            # Relative point position encoding: [center, neighbor, offset, dist] -> 10 dims
            rel_pos_encoding = torch.cat([center_xyz, neighbor_xyz, relative, dist], dim=-1)
            rel_pos_encoding = rel_pos_encoding.permute(0, 3, 1, 2)  # (B, 10, N, k)
            rel_pos_encoding = self.mlp(rel_pos_encoding)             # (B, C, N, k)

            feat_t = features.permute(0, 2, 1)                       # (B, N, C)
            neighbor_feat = index_points(feat_t, idx)                 # (B, N, k, C)
            neighbor_feat = neighbor_feat.permute(0, 3, 1, 2)          # (B, C, N, k)

            return torch.cat([rel_pos_encoding, neighbor_feat], dim=1)  # (B, 2C, N, k)

    class AttentivePooling(nn.Module):
        """Learned attention over the K neighbors, replacing max-pool —
        avoids discarding information the way a hard max would."""
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.score_fn = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
            )
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            """x: (B, C, N, k) -> (B, C_out, N)."""
            scores = self.score_fn(x)
            scores = F.softmax(scores, dim=-1)
            pooled = torch.sum(x * scores, dim=-1, keepdim=True)  # (B, C, N, 1)
            return self.mlp(pooled).squeeze(-1)                    # (B, C_out, N)

    class DilatedResidualBlock(nn.Module):
        """Two LocSE + AttentivePooling units with a skip connection —
        cheaply expands each point's receptive field to its two-hop
        neighborhood (K neighbors, then K^2 via feature propagation)."""
        def __init__(self, in_channel, out_channel, k):
            super().__init__()
            d = out_channel // 2
            self.fc1 = nn.Sequential(nn.Conv1d(in_channel, d, 1), nn.BatchNorm1d(d), nn.ReLU(inplace=True))

            self.locse1 = LocSE(d, k)
            self.ap1 = AttentivePooling(2 * d, d)

            self.locse2 = LocSE(d, k)
            self.ap2 = AttentivePooling(2 * d, out_channel)

            self.fc2 = nn.Sequential(nn.Conv1d(out_channel, out_channel, 1), nn.BatchNorm1d(out_channel))
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1), nn.BatchNorm1d(out_channel),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, xyz, features):
            """xyz:(B,N,3) features:(B,C_in,N) -> (B, C_out, N)."""
            x = self.fc1(features)
            x = self.locse1(xyz, x)
            x = self.ap1(x)
            x = self.locse2(xyz, x)
            x = self.ap2(x)
            x = self.fc2(x)
            return self.act(x + self.shortcut(features))

    return LocSE, AttentivePooling, DilatedResidualBlock


class RandLANetSegModel:
    """Real RandLA-Net segmentation network builder — per-point class logits.

    Example::

        model = RandLANetSegModel(num_classes=5, extra_feature_channels=1, k=16)
        logits = model(xyz)  # xyz: (B, 3+extra, N) -> (B, num_classes, N)
    """

    def __new__(cls, num_classes: int = 5, extra_feature_channels: int = 0,
                k: int = 16, decimation: int = 4, num_layers: int = 4):
        torch, nn, F = _require_torch()
        _, _, DRB = _build_lfa_modules()

        class RandLANetSeg(nn.Module):
            def __init__(self):
                super().__init__()
                in_ch = 3 + extra_feature_channels
                self.fc_start = nn.Sequential(
                    nn.Conv1d(in_ch, 8, 1), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
                )

                # Encoder: channel progression matching the original paper's
                # scale-down version (8 -> 32 -> 128 -> 256 -> 512).
                enc_channels = [8, 32, 128, 256, 512][: num_layers + 1]
                self.encoder = nn.ModuleList([
                    DRB(enc_channels[i], enc_channels[i + 1], k)
                    for i in range(num_layers)
                ])

                # Decoder: symmetric, nearest-neighbor upsample + skip concat + MLP
                dec_channels = enc_channels[::-1]
                self.decoder = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(dec_channels[i] + dec_channels[i + 1], dec_channels[i + 1], 1),
                        nn.BatchNorm1d(dec_channels[i + 1]), nn.ReLU(inplace=True),
                    )
                    for i in range(num_layers)
                ])

                self.fc_end = nn.Sequential(
                    nn.Conv1d(dec_channels[-1], 32, 1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Conv1d(32, num_classes, 1),
                )
                self.decimation = decimation

            def forward(self, xyz_full):
                # xyz_full: (B, 3+extra, N)
                B, _, N = xyz_full.shape
                xyz = xyz_full[:, :3, :].permute(0, 2, 1)  # (B, N, 3)
                features = self.fc_start(xyz_full)          # (B, 8, N)

                xyz_stack = [xyz]
                feat_stack = [features]

                cur_xyz, cur_feat = xyz, features
                for layer in self.encoder:
                    cur_feat = layer(cur_xyz, cur_feat)
                    n_next = max(cur_xyz.shape[1] // self.decimation, 1)
                    cur_xyz, cur_feat, _ = random_sample(cur_xyz, cur_feat.permute(0, 2, 1), n_next)
                    cur_feat = cur_feat.permute(0, 2, 1)
                    xyz_stack.append(cur_xyz)
                    feat_stack.append(cur_feat)

                # Decoder: upsample back through each encoder stage via
                # nearest-neighbor lookup + skip connection + MLP.
                for i, layer in enumerate(self.decoder):
                    target_xyz = xyz_stack[-(i + 2)]
                    idx = knn_point(1, cur_xyz, target_xyz).squeeze(-1)  # (B, S_target)
                    upsampled = index_points(cur_feat.permute(0, 2, 1), idx).permute(0, 2, 1)
                    skip = feat_stack[-(i + 2)]
                    cur_feat = layer(torch.cat([upsampled, skip], dim=1))
                    cur_xyz = target_xyz

                return self.fc_end(cur_feat)  # (B, num_classes, N)

        return RandLANetSeg()


def build_randlanet(num_classes: int = 5, in_channels: int = 3, k: int = 16, **kwargs) -> Any:
    """Factory used by the model registry — `get_model('randlanet', ...)`.
    `in_channels` = 3 (xyz only) + however many extra per-point features
    (intensity, RGB, return number, etc.)."""
    extra = max(in_channels - 3, 0)
    return RandLANetSegModel(num_classes=num_classes, extra_feature_channels=extra, k=k, **kwargs)