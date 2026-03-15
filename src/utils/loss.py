import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_dist
import numpy as np

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        """
        Adaptive Wing Loss for Heatmap Regression.
        
        Args:
            alpha (float): Curvature of the loss for small errors. 
                           Allows for smooth gradients near zero.
            omega (float): Controls the range of the non-linear part.
            epsilon (float): Curvature term.
            theta (float): Threshold between "linear" and "non-linear" response.
        """
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] Logits (will be sigmoid-ed) or raw probabilities
                  If your model outputs logits, use sigmoid first!
            target: [B, C, H, W] Gaussian Heatmaps (0.0 to 1.0)
        """
        # Apply sigmoid to logits to ensure 0-1 range
        y_pred = torch.sigmoid(pred) 
        y_true = target
        
        # Compute absolute difference
        diff = y_pred - y_true
        abs_diff = torch.abs(diff)
        
        # Adaptive Wing Loss Formula
        # Case 1: Error is large (> theta) -> Linear Loss (like L1)
        # Case 2: Error is small (<= theta) -> Non-linear Loss (focus on precision)
        
        # A & C constants for continuity at theta
        A = self.omega * (1 / (1 + (self.theta / self.epsilon)**(self.alpha - y_true))) * (self.alpha - y_true) * ((self.theta / self.epsilon)**(self.alpha - y_true - 1)) * (1 / self.epsilon)
        C = (self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon)**(self.alpha - y_true)))

        loss = torch.where(
            abs_diff < self.theta,
            self.omega * torch.log(1 + (abs_diff / self.epsilon)**(self.alpha - y_true)),
            A * abs_diff - C
        )
        weight = 1.0 + 4.0 * y_true
        return (loss * weight).mean()
    
class HeatmapAWingMSE(nn.Module):
    def __init__(self, wing_weight=1.0, mse_weight=0.1,
                 alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.awing = AdaptiveWingLoss(alpha, omega, epsilon, theta)
        self.mse_weight = mse_weight
        self.wing_weight = wing_weight

    def forward(self, pred, target):
        # pred: logits, target: heatmaps
        aw_loss = self.awing(pred, target)
        pred_sig = torch.sigmoid(pred)
        mse_loss = torch.mean((pred_sig - target)**2)
        return self.wing_weight * aw_loss + self.mse_weight * mse_loss