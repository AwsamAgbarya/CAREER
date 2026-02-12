import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_dist
import numpy as np

class SmoothnessLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        """
        Encourage piecewise smooth predictions within same class regions.
        
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] ground truth
        """
        # Convert to probabilities
        probs = F.softmax(pred, dim=1)  # [B, C, H, W]
        
        # Compute horizontal and vertical gradients
        grad_h = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])  # [B, C, H, W-1]
        grad_v = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])  # [B, C, H-1, W]
        
        # Penalize large gradients (class changes)
        smoothness = grad_h.mean() + grad_v.mean()
        
        return self.weight * smoothness
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        # Use log_softmax for numerical stability
        log_p = F.log_softmax(pred, dim=1)
        
        # Gather log probabilities for target classes
        log_p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute loss
        focal_loss = -focal_weight * log_p_t
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target.view(-1)).view_as(target)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        num_classes = pred.shape[1]
        pred_probs = F.softmax(pred, dim=1)
        
        # One-hot encode target: (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Vectorized intersection and union
        dims = (0, 2, 3)  # Reduce over batch, height, width
        intersection = (pred_probs * target_one_hot).sum(dim=dims)
        cardinality = pred_probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

def fast_sdf(target_mask, num_classes):
    b, h, w = target_mask.shape

    one_hot = F.one_hot(target_mask, num_classes).permute(0, 3, 1, 2).float()

    dist = torch.ones_like(one_hot) * 1000.0
    dist[one_hot == 1] = 0.0 # Distance is 0 at the hole pixels

    kernel = torch.ones(num_classes, 1, 3, 3).to(target_mask.device)
    kernel[:, 0, 1, 1] = 0

    foreground = (one_hot > 0.5).float()
    dist_map = torch.zeros_like(foreground)
    
    current_mask = foreground.clone()
    
    for i in range(20):
        dilated = F.conv2d(current_mask, kernel, padding=1, groups=num_classes)
        dilated = (dilated > 0).float()
        new_pixels = dilated - current_mask
        dist_map += new_pixels * (i + 1)
        current_mask = dilated

    dist_map[current_mask == 0] = 20.0
    return dist_map

class BoundaryLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        probs = F.softmax(pred, dim=1)
        
        # Calculate distance map (SDF)
        # Note: Ideally, pre-compute this in the DataLoader to speed up training!
        with torch.no_grad():
            gt_sdf = fast_sdf(target, num_classes).to(pred.device)
        
        # Loss = Probability * Distance
        # If model predicts high prob (1.0) at a pixel with distance 50px,
        # Loss = 1.0 * 50 = 50 (Huge penalty)
        loss = (probs * gt_sdf).mean()
        
        return self.weight * loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.3, focal_weight=0.3, 
                 boundary_weight=0.1, class_weights=None, focal_gamma=2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.class_weights = class_weights
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss(weight=1.0)
    
    def forward(self, pred, target):
        # # Standard losses
        # ce = F.cross_entropy(pred, target, weight=self.class_weights, ignore_index=0)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Boundary loss (forces shapes)
        boundary = self.boundary_loss(pred, target)
        
        total = (
                 self.dice_weight * dice + 
                 self.focal_weight * focal + 
                 self.boundary_weight * boundary)
        
        return total, {
            # 'ce': ce.detach().item(), 
            'dice': dice.detach().item(), 
            'focal': focal.detach().item(),
            'boundary': boundary.detach().item()
        }
