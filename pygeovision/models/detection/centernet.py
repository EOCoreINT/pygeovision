"""CenterNet ("Objects as Points", Zhou et al. 2019) -- an anchor-free detector.

Each object is a single peak in a class heat-map at its box centre; two further heads
regress the box size and the sub-pixel centre offset.  There are no anchors and no
non-maximum suppression: peaks are the 3x3 local maxima of the heat-map.

This is the "ResNet + deconvolution" variant from the paper (three stride-2
transposed convolutions, output stride 4), written here because neither torchvision
nor timm ships CenterNet.

API (torchvision-compatible, so ``GeoTrainer``'s detection mode drives it)::

    model = build_centernet(num_classes=3, in_channels=4, backbone="resnet18")
    model.train()
    losses = model(images, targets)          # {"loss_hm", "loss_wh", "loss_off"}
    model.eval()
    detections = model(images)               # [{"boxes", "scores", "labels"}, ...]

``targets[i]["boxes"]`` are ``(N, 4)`` ``xyxy`` pixel boxes and ``targets[i]["labels"]``
are integers in ``1..num_classes`` (0 is background, as in torchvision).  Detection
``labels`` use the same 1-based convention.  ``images`` is a list of ``(C, H, W)``
tensors or a ``(B, C, H, W)`` tensor, scaled to roughly [0, 1]; they are padded to a
multiple of 32 and normalised inside the model with per-channel mean/std (ImageNet
statistics for the first three bands; pass ``image_mean`` / ``image_std`` for your sensor).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pygeovision.models import _adapt

_STRIDE = 4


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Gaussian radius for a ``height`` x ``width`` object, as in the reference CenterNet code.

    This is CornerNet's (Law & Deng, 2018) three-case formula, reproduced exactly as the
    reference implementations compute it -- including their ``(b + sqrt(b^2 - 4ac)) / 2``
    (not ``/ 2a``) form -- so heat-map targets match the published training recipe.
    """
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))) / 2
    return min(r1, r2, r3)


def _draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    """Splat a 2-D Gaussian (peak 1 at the centre) onto ``heatmap`` with element-wise max."""
    diameter = 2 * radius + 1
    sigma = diameter / 6
    ys = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device)
    g1 = torch.exp(-(ys ** 2) / (2 * sigma * sigma))
    gauss = g1[:, None] * g1[None, :]
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    patch = gauss[radius - top:radius + bottom, radius - left:radius + right]
    torch.maximum(region, patch, out=region)


def _focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss of CornerNet/CenterNet (alpha=2, beta=4)."""
    pos = gt.eq(1).float()
    neg = 1 - pos
    neg_weight = torch.pow(1 - gt, 4)
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * neg_weight * neg).sum()
    num_pos = pos.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


class CenterNet(nn.Module):
    """ResNet + 3 deconvolution layers + (heat-map, size, offset) heads, output stride 4."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        backbone: str = "resnet18",
        pretrained: bool = False,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        score_thresh: float = 0.05,
        max_detections: int = 100,
        wh_weight: float = 0.1,
        off_weight: float = 1.0,
    ) -> None:
        super().__init__()
        import torchvision

        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        builders = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34,
                    "resnet50": torchvision.models.resnet50, "resnet101": torchvision.models.resnet101}
        if backbone not in builders:
            raise ValueError(f"backbone must be one of {sorted(builders)}, got {backbone!r}")
        net = builders[backbone](weights="DEFAULT" if pretrained else None)
        if in_channels != 3:
            net.conv1 = _adapt.adapt_first_conv(net.conv1, in_channels, pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
        feat = net.fc.in_features

        up, ch = [], feat
        for out_ch in (256, 128, 64):
            up += [nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            ch = out_ch
        self.deconv = nn.Sequential(*up)

        def head(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, out, 1))

        self.hm_head, self.wh_head, self.off_head = head(num_classes), head(2), head(2)
        self.hm_head[-1].bias.data.fill_(-2.19)  # start with ~10% foreground probability

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.max_detections = max_detections
        self.wh_weight, self.off_weight = wh_weight, off_weight
        mean = image_mean if image_mean is not None else _adapt.extend_stats(_adapt.IMAGENET_MEAN, in_channels)
        std = image_std if image_std is not None else _adapt.extend_stats(_adapt.IMAGENET_STD, in_channels)
        if len(mean) != in_channels or len(std) != in_channels:
            raise ValueError(f"image_mean/image_std need {in_channels} values each")
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        self.in_channels = in_channels

    # -- batching ------------------------------------------------------------------
    def _batch(self, images: Any) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if isinstance(images, torch.Tensor):
            images = list(images) if images.ndim == 4 else [images]
        if not images:
            raise ValueError("no images given")
        sizes = []
        for im in images:
            if im.ndim != 3 or im.shape[0] != self.in_channels:
                raise ValueError(
                    f"expected (C={self.in_channels}, H, W) images, got {tuple(im.shape)}")
            sizes.append((int(im.shape[1]), int(im.shape[2])))
        h = math.ceil(max(s[0] for s in sizes) / 32) * 32
        w = math.ceil(max(s[1] for s in sizes) / 32) * 32
        batch = images[0].new_zeros((len(images), self.in_channels, h, w))
        for i, im in enumerate(images):
            batch[i, :, :im.shape[1], :im.shape[2]] = im
        return (batch - self.mean) / self.std, sizes

    def _raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.deconv(self.layers(self.stem(x)))
        return self.hm_head(f), self.wh_head(f), self.off_head(f)

    # -- targets / loss -------------------------------------------------------------
    def _targets(self, targets: list[dict], shape: tuple[int, int], device: torch.device):
        oh, ow = shape
        b = len(targets)
        hm = torch.zeros(b, self.num_classes, oh, ow, device=device)
        wh = torch.zeros(b, 2, oh, ow, device=device)
        off = torch.zeros(b, 2, oh, ow, device=device)
        mask = torch.zeros(b, oh, ow, device=device)
        for i, t in enumerate(targets):
            boxes = t["boxes"].to(device).float()
            labels = t["labels"].to(device).long()
            if len(boxes) != len(labels):
                raise ValueError("boxes and labels differ in length")
            for box, lab in zip(boxes, labels):
                if not 1 <= int(lab) <= self.num_classes:
                    raise ValueError(
                        f"label {int(lab)} outside 1..{self.num_classes} (0 is background)")
                x1, y1, x2, y2 = (box / _STRIDE).tolist()
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ix, iy = min(max(int(cx), 0), ow - 1), min(max(int(cy), 0), oh - 1)
                radius = max(0, int(gaussian_radius(bh, bw)))
                _draw_gaussian(hm[i, int(lab) - 1], ix, iy, radius)
                wh[i, :, iy, ix] = torch.tensor([bw, bh], device=device)
                off[i, :, iy, ix] = torch.tensor([cx - ix, cy - iy], device=device)
                mask[i, iy, ix] = 1
        return hm, wh, off, mask

    def _loss(self, outs, targets, sizes):
        hm_p, wh_p, off_p = outs
        hm_t, wh_t, off_t, mask = self._targets(targets, hm_p.shape[-2:], hm_p.device)
        loss_hm = _focal_loss(torch.sigmoid(hm_p), hm_t)
        m = mask.unsqueeze(1)
        n = m.sum().clamp(min=1)
        loss_wh = (F.l1_loss(wh_p * m, wh_t * m, reduction="sum") / n) * self.wh_weight
        loss_off = (F.l1_loss(off_p * m, off_t * m, reduction="sum") / n) * self.off_weight
        return {"loss_hm": loss_hm, "loss_wh": loss_wh, "loss_off": loss_off}

    # -- decoding -------------------------------------------------------------------
    @torch.no_grad()
    def _decode(self, outs, sizes) -> list[dict[str, torch.Tensor]]:
        hm = torch.sigmoid(outs[0])
        peaks = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float() * hm
        b, c, oh, ow = peaks.shape
        k = min(self.max_detections, c * oh * ow)
        scores, idx = peaks.view(b, -1).topk(k)
        cls = idx // (oh * ow)
        pix = idx % (oh * ow)
        ys, xs = (pix // ow).float(), (pix % ow).float()
        wh = outs[1].view(b, 2, -1)
        off = outs[2].view(b, 2, -1)
        results = []
        for i in range(b):
            p = pix[i]
            w_, h_ = wh[i, 0, p], wh[i, 1, p]
            cx = (xs[i] + off[i, 0, p]) * _STRIDE
            cy = (ys[i] + off[i, 1, p]) * _STRIDE
            boxes = torch.stack([cx - w_ * _STRIDE / 2, cy - h_ * _STRIDE / 2,
                                 cx + w_ * _STRIDE / 2, cy + h_ * _STRIDE / 2], dim=1)
            ih, iw = sizes[i]
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, iw)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, ih)
            keep = (scores[i] >= self.score_thresh) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            results.append({"boxes": boxes[keep], "scores": scores[i][keep],
                            "labels": (cls[i][keep] + 1).long()})
        return results

    def forward(self, images: Any, targets: list[dict] | None = None):
        x, sizes = self._batch(images)
        outs = self._raw(x)
        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            if len(targets) != x.shape[0]:
                raise ValueError("need one target dict per image")
            return self._loss(outs, targets, sizes)
        return self._decode(outs, sizes)


def build_centernet(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    backbone: str = "resnet50",
    allow_random_init: bool = False,
    **kwargs: Any,
) -> CenterNet:
    """Build a :class:`CenterNet`.  ``num_classes`` counts object classes (no background).

    ``pretrained=True`` loads ImageNet backbone weights (the deconvolution layers and heads
    are always trained from scratch).  A failed download raises
    :class:`~pygeovision.core.exceptions.ModelLoadError` unless ``allow_random_init``.
    """
    from pygeovision.core.exceptions import ModelLoadError

    if not pretrained:
        return CenterNet(num_classes, in_channels, backbone, False, **kwargs)
    try:
        return CenterNet(num_classes, in_channels, backbone, True, **kwargs)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        if not allow_random_init:
            raise ModelLoadError(
                f"Could not load ImageNet weights for the CenterNet {backbone} backbone: {exc}",
                details="Pass pretrained=False for random initialisation, or "
                        "allow_random_init=True to fall back automatically.",
            ) from exc
        return _adapt.mark_random_init(
            CenterNet(num_classes, in_channels, backbone, False, **kwargs),
            f"pretrained load failed: {exc}")
