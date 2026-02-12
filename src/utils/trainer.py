import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import Subset
import bitsandbytes as bnb

from utils.loss import CombinedLoss
from utils.metrics import SegmentationMetrics, EMA

class SegmentationTrainer:
    
    def __init__(self, model, dataset, class_weights, config, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.num_classes = config.get('num_classes', 7)
        self.class_names = config['class_names']
        if class_weights is not None:
            self.class_weights = class_weights.to(device=device, dtype=torch.float32)
        
        # Setup directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.best_epoch = 0
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = config.get('epochs', 50)

        # Loss function - use combined loss
        self.criterion = CombinedLoss(
            ce_weight=config.get('ce_weight', 0.3),
            dice_weight=config.get('dice_weight', 0.3),
            focal_weight=config.get('focal_weight', 0.3),
            boundary_weight=config.get('boundary_weight', 0.1),
            class_weights=self.class_weights,
            focal_gamma=config.get('focal_gamma', 2.0),
        )
        
        # Optimizer - differential learning rates
        backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
        decoder_params = [p for n, p in model.named_parameters() if 'seg_head' in n]
        self.optimizer = bnb.optim.AdamW8bit([
            {'params': backbone_params, 'lr': config.get('backbone_lr', 1e-5)},
            {'params': decoder_params, 'lr': config.get('decoder_lr', 1e-4)}
        ], weight_decay=config.get('weight_decay', 1e-4))
        
        # Learning rate scheduler
        warmup_scheduler = LinearLR(self.optimizer, start_factor=config.get('start_lr', 0.01), total_iters=warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        # Mixed precision training
        self.scaler = GradScaler(device='cuda')
        # self.ema = EMA(model, decay=config.get('ema_decay', 0.999))
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=self.num_classes, class_names=self.class_names)
        
        # Tracking
        self.best_miou = 0.0
        self.train_losses = []
        self.val_metrics = []

        val_split = config.get('val_split', 0.2)
        val_size = int(len(dataset) * val_split)
        indices = list(range(len(dataset)))
        
        # Store indices as instance variables
        self.train_indices = indices[:-val_size]
        self.val_indices = indices[-val_size:]
        
        # Create DataLoaders once
        train_dataset = Subset(dataset, self.train_indices)
        val_dataset = Subset(dataset, self.val_indices)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            persistent_workers=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            persistent_workers=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
        )
            
        # W&B logging
        if config.get('use_wandb', False):
            wandb.init(project=config.get('wandb_project', 'vmamba-seg'), config=config)
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0.0
        loss_components = defaultdict(float)
        accumulation_steps = self.config.get('accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (img, mask) in enumerate(pbar):
            img = img.to(self.device, non_blocking=True)
            mask = mask.squeeze(1).long().to(self.device, non_blocking=True)

            # Forward + backward
            with autocast(device_type='cuda'):
                seg_logits = self.model(img)
                loss, loss_dict = self.criterion(seg_logits, mask)
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # Update every N steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # self.ema.update()
            
            # Logging
            loss_val = loss.item() * accumulation_steps
            total_loss += loss_val
            for key, val in loss_dict.items():
                loss_components[key] += val
            
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        # Handle remaining gradients if batch count not divisible
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v / len(self.train_loader) for k, v in loss_components.items()}
        }

    
    def validate(self):
        """Validate on validation split."""
        # self.ema.apply_shadow()
        self.model.eval()
        checkpoint_enabled = False
        if hasattr(self.model.backbone, 'gradient_checkpointing'):
            checkpoint_enabled = self.model.backbone.gradient_checkpointing
            self.model.backbone.gradient_checkpointing = False

        self.metrics.reset()
        total_loss = 0.0
        all_keypoint_metrics = defaultdict(list)
        
        pbar = tqdm(self.val_loader, desc='Validation')
        with torch.no_grad():
            for img, mask in pbar:
                img = img.to(self.device, non_blocking=True)
                mask = mask.squeeze(1).long().to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda'):
                    seg_logits = self.model(img)
                    loss, _ = self.criterion(seg_logits, mask)
                
                total_loss += loss.item()
                self.metrics.update(seg_logits, mask)
                
                kp_metrics = self.compute_keypoint_metrics(seg_logits, mask)
                for key, val in kp_metrics.items():
                    all_keypoint_metrics[key].append(val)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics_summary = self.metrics.get_summary()
        metrics_summary['val_loss'] = avg_loss
        
        for key, values in all_keypoint_metrics.items():
            metrics_summary[key] = np.mean(values)
        
        if hasattr(self.model.backbone, 'gradient_checkpointing'):
            self.model.backbone.gradient_checkpointing = checkpoint_enabled

        # self.ema.restore()
        return metrics_summary
        
    def save_checkpoint(self, epoch, metrics, is_best):
        """Save latest checkpoint and best model based on mIoU."""
        miou = metrics['mIoU']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': metrics,
            'config': self.config
        }
        
        # Always save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint if this is the best mIoU so far
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"✓ Saved best checkpoint (mIoU: {miou:.4f})")
    
    def train(self):
        """Main training loop."""
        epochs = self.config.get('epochs', 50)
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate step
            self.scheduler.step()
            
            # Print summary
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Components: {train_metrics}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"mIoU: {val_metrics['mIoU']:.4f} | mF1: {val_metrics['mF1']:.4f}")
            for class_name in ['hole_1', 'hole_2', 'hole_3', 'hole_4', 'hole_5', 'center']:
                print(f"  {class_name:12} - IoU: {val_metrics[f'{class_name}_IoU']:.4f}, "
                    f"F1: {val_metrics[f'{class_name}_F1']:.4f}")
            
            is_best = val_metrics['mIoU'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mIoU']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint (pass is_best flag to handle saving best.pt)
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch+1}")
                break
            
            # Visualization
            if (epoch + 1) % self.config.get('vis_frequency', 5) == 0:
                self.visualize_predictions(epoch)
            
            # W&B logging
            if self.config.get('use_wandb', False):
                wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
            
            self.val_metrics.append(val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch+1}")
                break
        
        print(f"\nTraining complete!")
        print(f"Best checkpoint: {self.checkpoint_dir / 'best.pt'} (mIoU: {self.best_miou:.4f})")
        print(f"Latest checkpoint: {self.checkpoint_dir / 'latest.pt'}")

    def visualize_predictions(self, epoch):
        """Save visualization samples during training."""
        self.model.eval()
        
        # Use first 4 validation samples
        vis_indices = self.val_indices[:4]
        
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(vis_indices):
                img, mask = self.dataset[sample_idx]
                img_batch = img.unsqueeze(0).to(self.device)
                
                pred = self.model(img_batch)
                pred_mask = pred.argmax(dim=1).squeeze().cpu()
                
                # Plot
                axes[idx, 0].imshow(img.permute(1, 2, 0).cpu())
                axes[idx, 0].set_title('Input')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(mask.squeeze().cpu(), cmap='tab10', vmin=0, vmax=5)
                axes[idx, 1].set_title('Ground Truth')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=5)
                axes[idx, 2].set_title('Prediction')
                axes[idx, 2].axis('off')
        
        plt.tight_layout()
        save_path = self.checkpoint_dir / f'vis_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.config.get('use_wandb', False):
            wandb.log({"predictions": wandb.Image(str(save_path))}, step=epoch)


    def compute_keypoint_metrics(self, predictions, targets):
        """Compute detection rate and localization error for keypoints."""
        pred_masks = predictions.argmax(dim=1)  # [B, H, W]
        batch_size = pred_masks.shape[0]
        num_classes = predictions.shape[1]
        
        metrics = {}
        
        # Vectorized computation per class
        for class_id in range(1, num_classes):  # Skip background
            # Get all masks at once
            pred_mask = (pred_masks == class_id)  # [B, H, W]
            gt_mask = (targets == class_id)  # [B, H, W]
            
            # Samples with GT keypoint present
            has_gt = gt_mask.sum(dim=(1, 2)) > 0  # [B]
            num_gt_samples = has_gt.sum().item()
            
            if num_gt_samples == 0:
                continue
            
            # Detection: predicted where GT exists
            has_pred = pred_mask.sum(dim=(1, 2)) > 0  # [B]
            detected = (has_gt & has_pred).sum().item()
            detection_rate = detected / num_gt_samples
            
            # Localization error (only for detected)
            errors = []
            for b in range(batch_size):
                if not (has_gt[b] and has_pred[b]):
                    continue
                
                # Compute centroids
                pred_coords = torch.nonzero(pred_mask[b], as_tuple=False).float()
                gt_coords = torch.nonzero(gt_mask[b], as_tuple=False).float()
                
                pred_centroid = pred_coords.mean(dim=0)  # [y, x]
                gt_centroid = gt_coords.mean(dim=0)
                
                error = torch.dist(pred_centroid, gt_centroid).item()
                errors.append(error)
            
            avg_error = np.mean(errors) if errors else 0.0
            
            metrics[f'kp{class_id}_detection_rate'] = detection_rate
            metrics[f'kp{class_id}_error_px'] = avg_error
        
        # Convert to tensors for proper averaging
        if metrics:
            detection_rates = [v for k, v in metrics.items() if 'detection' in k]
            errors = [v for k, v in metrics.items() if 'error' in k and v > 0]
            
            metrics['avg_detection_rate'] = np.mean(detection_rates)
            metrics['avg_localization_error_px'] = np.mean(errors) if errors else 0.0
        
        return metrics

