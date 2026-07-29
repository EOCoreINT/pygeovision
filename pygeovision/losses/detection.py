"""IoU-family losses for object detection (D1)."""
from __future__ import annotations

import math
from typing import Any


class CIoULoss:
    """Complete IoU loss — accounts for overlap, distance, and aspect ratio."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)
            inter_x = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
            inter_y = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
            inter = inter_x * inter_y
            pred_area   = (px2-px1).clamp(0) * (py2-py1).clamp(0)
            target_area = (tx2-tx1).clamp(0) * (ty2-ty1).clamp(0)
            union = pred_area + target_area - inter
            iou = inter / union.clamp(min=1e-6)
            # Enclosing box diagonal
            enc_x = torch.max(px2,tx2) - torch.min(px1,tx1)
            enc_y = torch.max(py2,ty2) - torch.min(py1,ty1)
            c2 = enc_x**2 + enc_y**2 + 1e-6
            # Centre distance
            pc_x, pc_y = (px1+px2)/2, (py1+py2)/2
            tc_x, tc_y = (tx1+tx2)/2, (ty1+ty2)/2
            d2 = (pc_x-tc_x)**2 + (pc_y-tc_y)**2
            # Aspect ratio term
            pw, ph = (px2-px1).clamp(1e-6), (py2-py1).clamp(1e-6)
            tw, th = (tx2-tx1).clamp(1e-6), (ty2-ty1).clamp(1e-6)
            v = (4/math.pi**2) * (torch.atan(tw/th) - torch.atan(pw/ph))**2
            with torch.no_grad():
                alpha = v / (1 - iou + v + 1e-6)
            ciou = iou - d2/c2 - alpha*v
            return (1 - ciou).mean()
        except ImportError:
            raise ImportError("torch required")

class DIoULoss:
    """Distance IoU loss — penalises center-point distance on top of IoU,
    without CIoU's additional aspect-ratio term."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)
            inter_x = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
            inter_y = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
            inter = inter_x * inter_y
            pred_area   = (px2-px1).clamp(0) * (py2-py1).clamp(0)
            target_area = (tx2-tx1).clamp(0) * (ty2-ty1).clamp(0)
            union = pred_area + target_area - inter
            iou = inter / union.clamp(min=1e-6)
            # Enclosing box diagonal
            enc_x = torch.max(px2,tx2) - torch.min(px1,tx1)
            enc_y = torch.max(py2,ty2) - torch.min(py1,ty1)
            c2 = enc_x**2 + enc_y**2 + 1e-6
            # Centre distance
            pc_x, pc_y = (px1+px2)/2, (py1+py2)/2
            tc_x, tc_y = (tx1+tx2)/2, (ty1+ty2)/2
            d2 = (pc_x-tc_x)**2 + (pc_y-tc_y)**2
            diou = iou - d2/c2
            return (1 - diou).mean()
        except ImportError:
            raise ImportError("torch required")

class GIoULoss:
    """Generalised IoU loss."""
    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)
            inter = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) * (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
            pa = (px2-px1).clamp(0)*(py2-py1).clamp(0)
            ta = (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0)
            union = pa+ta-inter
            iou = inter/union.clamp(1e-6)
            enc = (torch.max(px2,tx2)-torch.min(px1,tx1)).clamp(0)*(torch.max(py2,ty2)-torch.min(py1,ty1)).clamp(0)
            giou = iou - (enc-union)/enc.clamp(1e-6)
            return (1-giou).mean()
        except ImportError:
            raise ImportError("torch required")

class SIoULoss:
    """SIoU (Scylla-IoU) loss — Gevorgyan (2022), "SIoU Loss: More Powerful
    Learning for Bounding Box Regression". Adds an angle-aware distance
    cost (drives predictions toward the nearest axis first, reducing
    degrees of freedom) plus a shape cost, on top of IoU. Distinct from
    GIoU/CIoU/DIoU — has its own four cost components (angle, distance,
    shape, IoU), not a variant of any of them.
    """
    def __init__(self, theta: float = 4.0) -> None:
        self.theta = theta  # shape-cost attention parameter (paper: range 2-6, ~4 typical)

    def __call__(self, pred_boxes: Any, target_boxes: Any) -> Any:
        try:
            import torch
            px1,py1,px2,py2 = pred_boxes.unbind(-1)
            tx1,ty1,tx2,ty2 = target_boxes.unbind(-1)

            inter_x = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
            inter_y = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
            inter = inter_x * inter_y
            pw, ph = (px2-px1).clamp(1e-6), (py2-py1).clamp(1e-6)
            tw, th = (tx2-tx1).clamp(1e-6), (ty2-ty1).clamp(1e-6)
            pred_area, target_area = pw*ph, tw*th
            union = pred_area + target_area - inter
            iou = inter / union.clamp(min=1e-6)

            pc_x, pc_y = (px1+px2)/2, (py1+py2)/2
            tc_x, tc_y = (tx1+tx2)/2, (ty1+ty2)/2
            enc_w = torch.max(px2,tx2) - torch.min(px1,tx1) + 1e-6
            enc_h = torch.max(py2,ty2) - torch.min(py1,ty1) + 1e-6

            # Angle cost: drives the prediction toward the nearer axis first.
            # NOTE: epsilon must go INSIDE the sqrt, not as a clamp on its
            # output — clamping sqrt(x)'s output does not protect its
            # gradient (d/dx sqrt(x) = 1/(2*sqrt(x)), which is still NaN at
            # x=0 regardless of a downstream clamp). Verified this produces
            # a finite gradient even for identical/coincident boxes.
            sigma = torch.sqrt((tc_x-pc_x)**2 + (tc_y-pc_y)**2 + 1e-9)
            c_h = (tc_y - pc_y).abs()
            sin_alpha = (c_h / sigma).clamp(-1 + 1e-6, 1 - 1e-6)
            angle_cost = 1 - 2 * torch.sin(torch.arcsin(sin_alpha) - math.pi/4) ** 2

            # Distance cost: angle-modulated normalised center-distance penalty
            gamma = 2 - angle_cost
            rho_x = ((tc_x - pc_x) / enc_w) ** 2
            rho_y = ((tc_y - pc_y) / enc_h) ** 2
            distance_cost = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))

            # Shape cost: aspect-ratio mismatch, weighted by theta
            omega_w = (pw - tw).abs() / torch.max(pw, tw)
            omega_h = (ph - th).abs() / torch.max(ph, th)
            shape_cost = (1 - torch.exp(-omega_w)) ** self.theta + (1 - torch.exp(-omega_h)) ** self.theta

            loss = 1 - iou + (distance_cost + shape_cost) / 2
            return loss.mean()
        except ImportError:
            raise ImportError("torch required")