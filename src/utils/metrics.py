import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class SegmentationMetrics:
    def __init__(self, num_classes=7, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        # Use torch tensor for GPU acceleration
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
    
    def update(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        pred_labels = torch.argmax(pred, dim=1)  # (B, H, W)
        
        # Keep on GPU, flatten
        pred_flat = pred_labels.view(-1)
        target_flat = target.view(-1)
        valid_mask = (target_flat >= 0) & (target_flat < self.num_classes) & \
                     (pred_flat >= 0) & (pred_flat < self.num_classes)
        
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        # Create indices: target * num_classes + pred
        indices = target_flat * self.num_classes + pred_flat
        bincount = torch.bincount(indices, minlength=self.num_classes ** 2)
        
        # Reshape and add to confusion matrix
        bincount = bincount[:self.num_classes ** 2]  # Truncate if longer
        confusion_update = bincount.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += confusion_update.cpu()
    
    def compute_iou(self):
        """Compute per-class and mean IoU."""
        # Diagonal = True Positives
        tp = np.diag(self.confusion_matrix.numpy())
        fp = self.confusion_matrix.numpy().sum(axis=0) - tp
        fn = self.confusion_matrix.numpy().sum(axis=1) - tp
        
        iou = tp / (tp + fp + fn + 1e-8)
        return iou, iou.mean()

    def compute_f1(self):
        """Compute per-class and mean F1 score."""
        tp = np.diag(self.confusion_matrix.numpy())
        fp = self.confusion_matrix.numpy().sum(axis=0) - tp
        fn = self.confusion_matrix.numpy().sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return f1, f1.mean()
    
    def compute_accuracy(self):
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix.numpy()).sum()
        total = self.confusion_matrix.numpy().sum()
        return correct / (total + 1e-8)
    
    def get_summary(self):
        """Return formatted summary."""
        iou_per_class, miou = self.compute_iou()
        f1_per_class, mf1 = self.compute_f1()
        acc_per_class_mean = self.compute_accuracy()
        
        summary = {'mIoU': miou, 'mF1': mf1, 'mAcc':acc_per_class_mean}
        for i, name in enumerate(self.class_names):
            summary[f'{name}_IoU'] = iou_per_class[i]
            summary[f'{name}_F1'] = f1_per_class[i]
        
        return summary


class KeypointMetrics:
    def __init__(self, num_classes=6, class_names=None, threshold=5.0):
        """
        Args:
            threshold (float): Distance threshold (pixels) to consider a detection 'correct'.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        # Store errors for each class to compute means later
        self.errors = {i: [] for i in range(self.num_classes)}
        # Store hits (correct detections) vs total attempts
        self.detections = {i: 0 for i in range(self.num_classes)}
        self.counts = {i: 0 for i in range(self.num_classes)}
        
    def get_coords_from_heatmap(self, heatmap):
        """
        Find (x, y) coordinates of peak in heatmap.
        heatmap: [B, C, H, W]
        Returns: [B, C, 2] coordinates (x, y)
        """
        batch_size, num_classes, h, w = heatmap.shape
        # Flatten H,W to find max index
        flat_map = heatmap.view(batch_size, num_classes, -1)
        max_vals, max_indices = torch.max(flat_map, dim=2)
        
        # Convert index back to (x, y)
        y = (max_indices // w).float()
        x = (max_indices % w).float()
        
        coords = torch.stack([x, y], dim=2) # [B, C, 2]
        return coords, max_vals

    def update(self, pred_logits, target_heatmaps):
        """
        Args:
            pred_logits: (B, C, H, W) Logits from model
            target_heatmaps: (B, C, H, W) GT Heatmaps (0..1)
        """
        # Apply Sigmoid to prediction
        pred_maps = torch.sigmoid(pred_logits)
        
        # Get Coordinates of peaks
        pred_coords, pred_confs = self.get_coords_from_heatmap(pred_maps)
        gt_coords, gt_confs = self.get_coords_from_heatmap(target_heatmaps)
        
        # Compute Metrics
        batch_size = pred_coords.shape[0]
        
        for b in range(batch_size):
            for c in range(self.num_classes): # Skip background (Index 0) usually
                if gt_confs[b, c] > 0.5:
                    # Euclidean Distance
                    dist = torch.norm(pred_coords[b, c] - gt_coords[b, c]).item()
                    
                    self.errors[c].append(dist)
                    self.counts[c] += 1
                    
                    if dist <= self.threshold:
                        self.detections[c] += 1

    def get_summary(self):
        summary = {}
        total_error = []
        total_detections = 0
        total_counts = 0
        
        for c in range(self.num_classes): # Skip Background
            name = self.class_names[c]
            
            # Mean Radial Error (Pixel Distance)
            if self.errors[c]:
                mre = np.mean(self.errors[c])
                summary[f'{name}_MRE'] = mre
                total_error.extend(self.errors[c])
            else:
                summary[f'{name}_MRE'] = 0.0
            
            # Detection Rate (PCK)
            if self.counts[c] > 0:
                pck = self.detections[c] / self.counts[c]
                summary[f'{name}_PCK'] = pck
                
                total_detections += self.detections[c]
                total_counts += self.counts[c]
            else:
                summary[f'{name}_PCK'] = 0.0
        
        # Global Metrics
        summary['Mean_Error'] = np.mean(total_error) if total_error else 0.0
        summary['Mean_PCK'] = total_detections / total_counts if total_counts > 0 else 0.0
        summary['mIoU'] = summary['Mean_PCK'] 
        
        return summary

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Temporarily apply EMA weights to model."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

def compute_class_weights(dataset, num_classes=3, num_workers=4):
    """Compute balanced class weights with better normalization."""
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=8, num_workers=num_workers, shuffle=False)
    
    print("Computing class weights...")
    for _, masks in tqdm(loader):
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum()
    
    total_pixels = class_counts.sum().float()
    weights = total_pixels / (num_classes * class_counts.float())
    
    return weights
