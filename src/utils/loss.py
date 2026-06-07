import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_dist
import numpy as np
class DistanceAwareCrossChannelLoss(nn.Module):
    """
    Penalizes channel c for activating near channel j's GT peak,
    but ONLY in regions that are far from channel c's own GT peak.
    Allows natural Gaussian overlap at keypoint boundaries.
    """
    def __init__(self, sigma, distance_factor=2):
        super().__init__()
        self.sigma = sigma            # same sigma used to generate GT heatmaps
        self.distance_factor = distance_factor  # multiples of sigma = safe radius

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        device = pred.device

        # Build spatial coordinate grid once
        gy = torch.arange(H, device=device).float().view(1, 1, H, 1)
        gx = torch.arange(W, device=device).float().view(1, 1, 1, W)

        # Extract GT peak locations (argmax of each GT channel)
        gt_flat = target.view(B, C, -1)
        peak_idx = gt_flat.argmax(dim=-1)           # [B, C]
        peak_y = (peak_idx // W).float()            # [B, C]
        peak_x = (peak_idx % W).float()             # [B, C]

        safe_radius = self.distance_factor * self.sigma  # pixels
        loss = torch.tensor(0.0, device=device)

        for c in range(C):
            # Distance from every pixel to channel c's own GT peak → safe zone
            own_py = peak_y[:, c].view(B, 1, 1)    # [B, 1, 1]
            own_px = peak_x[:, c].view(B, 1, 1)
            dist_own = torch.sqrt(
                (gy.squeeze(1) - own_py) ** 2 +
                (gx.squeeze(0) - own_px) ** 2 + 1e-6
            )  # [B, H, W]

            # Soft "far from own peak" mask — smoothly fades in past safe_radius
            far_mask = 1.0 - torch.exp(-((dist_own - safe_radius).clamp(min=0)) ** 2
                                        / (2 * self.sigma ** 2))  # [B, H, W]

            for j in range(C):
                if j == c:
                    continue

                # Soft "near channel j's peak" mask
                other_py = peak_y[:, j].view(B, 1, 1)
                other_px = peak_x[:, j].view(B, 1, 1)
                dist_other = torch.sqrt(
                    (gy.squeeze(1) - other_py) ** 2 +
                    (gx.squeeze(0) - other_px) ** 2 + 1e-6
                )
                near_other_mask = torch.exp(-dist_other ** 2 / (2 * self.sigma ** 2))  # [B, H, W]

                # Penalty zone: far from own peak AND near another channel's peak
                penalty_mask = far_mask * near_other_mask   # [B, H, W]

                # Penalize pred channel c activating in this zone
                loss = loss + (pred[:, c] * penalty_mask).mean()

        return loss / (C * (C - 1))
    
class PeakSharpnessLoss(nn.Module):
    def __init__(self, sigma, concentration_factor=4.0, min_gt_peak=0.3):
        super().__init__()
        self.sigma = sigma
        self.concentration_factor = concentration_factor
        self.min_gt_peak = min_gt_peak

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        device = pred.device

        gy = torch.arange(H, device=device).float().view(1, 1, H, 1)
        gx = torch.arange(W, device=device).float().view(1, 1, 1, W)

        gt_flat = target.view(B, C, -1)
        peak_idx = gt_flat.argmax(dim=-1)
        peak_y = (peak_idx // W).float()
        peak_x = (peak_idx % W).float()
        gt_peak_val = gt_flat.max(dim=-1)[0]  # [B, C] — max GT value per channel

        loss = torch.tensor(0.0, device=device)
        focus_sigma = self.concentration_factor * self.sigma
        count = 0

        for c in range(C):
            # Only apply sharpness where the GT says keypoint is actually present
            present = gt_peak_val[:, c] > self.min_gt_peak  # [B] bool mask
            if not present.any():
                continue

            py = peak_y[:, c].view(B, 1, 1)
            px = peak_x[:, c].view(B, 1, 1)
            dist_sq = (gy.squeeze(1) - py) ** 2 + (gx.squeeze(0) - px) ** 2

            inside_weight = torch.exp(-dist_sq / (2 * focus_sigma ** 2))
            outside_weight = 1.0 - inside_weight

            pred_c = pred[:, c]  # [B, H, W]
            total_act = pred_c.sum(dim=(-1, -2)).clamp(min=0.5)
            outside_energy = (pred_c * outside_weight).sum(dim=(-1, -2))  # [B]

            ratio = outside_energy / total_act  # [B]
            loss = loss + ratio[present].mean()
            count += 1

        return loss / max(count, 1)

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, y_pred, target):
        y_true = target
        abs_diff = torch.abs(y_pred - y_true)
        A = (self.omega
             * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y_true)))
             * (self.alpha - y_true)
             * ((self.theta / self.epsilon) ** (self.alpha - y_true - 1))
             / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - y_true))
        loss = torch.where(
            abs_diff < self.theta,
            self.omega * torch.log(1 + (abs_diff / self.epsilon) ** (self.alpha - y_true)),
            A * abs_diff - C
        )
        return (loss * (1.0 + 4.0 * y_true)).mean()


class HeatmapAWingMSE(nn.Module):
    # Exposed so the trainer can auto-discover component names
    loss_keys = ['awing', 'mse', 'cross', 'sharp']

    def __init__(self, sigma, wing_weight=1.0, mse_weight=0.1,
                 cross_weight=0.1, sharp_weight=0.05,
                 distance_factor=2.5,
                 alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.awing = AdaptiveWingLoss(alpha, omega, epsilon, theta)
        self.cross_channel = DistanceAwareCrossChannelLoss(sigma, distance_factor)
        self.sharpness = PeakSharpnessLoss(sigma=sigma,)
        self.wing_weight = wing_weight
        self.mse_weight = mse_weight
        self.sharp_weight = sharp_weight
        self.cross_weight = cross_weight

    def forward(self, pred, target):
        """
        Returns a dict with keys: 'total' (tensor, use for .backward()),
        and 'awing', 'mse', 'cross', 'sharp' (detached tensors, use for logging).
        """
        aw_loss    = self.awing(pred, target)
        mse_loss   = torch.mean((pred - target) ** 2)
        cross_loss = self.cross_channel(pred, target)
        sharp_loss = self.sharpness(pred, target)

        total = (self.wing_weight * aw_loss
               + self.mse_weight  * mse_loss
               + cross_loss * self.cross_weight
               + sharp_loss * self.sharp_weight)

        return {
            'total': total,
            'awing': aw_loss.detach(),
            'mse':   mse_loss.detach(),
            'cross': cross_loss.detach(),
            'sharp': sharp_loss.detach(),
        }