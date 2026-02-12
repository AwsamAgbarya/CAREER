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
