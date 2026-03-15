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
import json

# Import your updated modules
from utils.loss import HeatmapAWingMSE
from utils.metrics import KeypointMetrics

class KeypointTrainer:
    
    def __init__(self, model, dataset, config, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.num_classes = config.get('num_classes', 6)
        self.class_names = config.get('class_names', [f'kp_{i}' for i in range(self.num_classes)])
        
        # Setup directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.best_epoch = 0
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = config.get('epochs', 50)

        # Loss function - Adaptive Wing Loss for Heatmaps
        self.criterion = HeatmapAWingMSE(
            wing_weight=config.get('w_weight', 2.1),
            mse_weight=config.get('mse_weight', 2.1),
            alpha=config.get('alpha', 2.1),
            omega=config.get('omega', 14.0),
            epsilon=config.get('epsilon', 1.0),
            theta=config.get('theta', 0.5)
        )
        
        # Optimizer
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
        
        # Mixed precision
        self.scaler = GradScaler(device='cuda')
        
        # Metrics - Using Keypoint Metrics (PCK, Mean Error)
        self.metrics = KeypointMetrics(
            num_classes=self.num_classes, 
            class_names=self.class_names,
            threshold=config.get('metric_threshold', 5.0) # 5 pixel threshold
        )

        # Tracking
        self.best_pck = 0.0 # Track PCK instead of mIoU
        self.train_losses = []
        self.val_metrics = []

                # Histories for plotting
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "mean_pck": [],
            "mean_error": [],
        }
        # Per-class histories (dict of name -> list)
        self.class_history = {
            name: {"PCK": [], "MRE": []}
            for name in self.class_names
        }


        val_split = config.get('val_split', 0.2)
        val_size = int(len(dataset) * val_split)
        indices = list(range(len(dataset)))
        
        self.train_indices = indices[:-val_size]
        self.val_indices = indices[-val_size:]
        
        # Create DataLoaders
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
            shuffle=True,
            persistent_workers=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True,
        )
            
        if config.get('use_wandb', False):
            wandb.init(project=config.get('wandb_project', 'vmamba-keypoints'), config=config)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        accumulation_steps = self.config.get('accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (img, heatmaps) in enumerate(pbar):
            img = img.to(self.device, non_blocking=True)
            # heatmaps are [B, C, H, W] float tensors
            heatmaps = heatmaps.to(self.device, non_blocking=True)

            # Forward + backward
            with autocast(device_type='cuda'):
                logits = self.model(img) # [B, C, H, W] logits
                loss = self.criterion(logits, heatmaps)
                loss = loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            loss_val = loss.item() * accumulation_steps
            total_loss += loss_val 
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate(self):
        """Validate on validation split."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        with torch.no_grad():
            for img, heatmaps in pbar:
                img = img.to(self.device, non_blocking=True)
                heatmaps = heatmaps.to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda'):
                    logits = self.model(img)
                    loss = self.criterion(logits, heatmaps)
                
                total_loss += loss.item()
                # Update metrics (Logits -> Sigmoid -> Coords)
                self.metrics.update(logits, heatmaps)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics_summary = self.metrics.get_summary()
        metrics_summary['val_loss'] = avg_loss
        
        return metrics_summary
        
    def save_checkpoint(self, epoch, metrics, is_best):
        """Save latest checkpoint and best model based on Mean PCK."""
        pck = metrics.get('Mean_PCK', 0.0)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_pck': self.best_pck,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"✓ Saved best checkpoint (PCK: {pck:.4f})")
    
    def train(self):
        epochs = self.config.get('epochs', 50)
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            
            # Print summary
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Mean PCK: {val_metrics['Mean_PCK']:.4f} | Mean Error: {val_metrics['Mean_Error']:.2f} px")
            # Save history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["mean_pck"].append(val_metrics.get("Mean_PCK", 0.0))
            self.history["mean_error"].append(val_metrics.get("Mean_Error", 0.0))
            
            # Print per-class metrics if available
            for class_name in self.class_names:
                pck_key = f"{class_name}_PCK"
                mre_key = f"{class_name}_MRE"
                self.class_history[class_name]["PCK"].append(val_metrics.get(pck_key, 0.0))
                self.class_history[class_name]["MRE"].append(val_metrics.get(mre_key, 0.0))
                if f'{class_name}_PCK' in val_metrics:
                    print(f"  {class_name:12} - PCK: {val_metrics[f'{class_name}_PCK']:.4f}, "
                          f"Err: {val_metrics[f'{class_name}_MRE']:.2f} px")
            
            # Early Stopping Check
            current_pck = val_metrics.get('Mean_PCK', 0.0)
            is_best = current_pck > self.best_pck
            
            if is_best:
                self.best_pck = current_pck
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            if (epoch + 1) % self.config.get('vis_frequency', 10) == 0:
                self.visualize_predictions(epoch)
            
            if self.config.get('use_wandb', False):
                wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
            
            self.val_metrics.append(val_metrics)
            
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best PCK: {self.best_pck:.4f} at epoch {self.best_epoch+1}")
                break
        
        self._save_history()
        self._plot_curves()
        print(f"\nTraining complete!")
        print(f"Best checkpoint: {self.checkpoint_dir / 'best.pt'}")

    def visualize_predictions(self, epoch):
        """Save visualization samples (Input, GT Heatmap, Pred Heatmap)."""
        self.model.eval()
        vis_indices = self.val_indices[:4]
        
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(vis_indices):
                # Dataset returns img, heatmap
                img, gt_heatmap = self.dataset[sample_idx] 
                img_batch = img.unsqueeze(0).to(self.device)
                
                logits = self.model(img_batch)
                pred_heatmap = torch.sigmoid(logits).squeeze(0).cpu() # [C, H, W]
                gt_heatmap = gt_heatmap.cpu() # [C, H, W]
                
                
                gt_vis = torch.max(gt_heatmap, dim=0)[0]
                pred_vis = torch.max(pred_heatmap, dim=0)[0]
                
                # Plot
                axes[idx, 0].imshow(img.permute(1, 2, 0).cpu())
                axes[idx, 0].set_title('Input')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(gt_vis, cmap='hot', vmin=0, vmax=1)
                axes[idx, 1].set_title('GT Heatmap')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(pred_vis, cmap='hot', vmin=0, vmax=1)
                axes[idx, 2].set_title('Pred Heatmap')
                axes[idx, 2].axis('off')
        
        plt.tight_layout()
        save_path = self.checkpoint_dir / f'vis_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if self.config.get('use_wandb', False):
            wandb.log({"predictions": wandb.Image(str(save_path))}, step=epoch)

    def _save_history(self):
        """Save raw metrics history to JSON."""
        hist_path = self.checkpoint_dir / "metrics_history.json"
        # Convert numpy/torch types to Python native
        history = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        class_history = {
            name: {
                "PCK": [float(v) for v in vals["PCK"]],
                "MRE": [float(v) for v in vals["MRE"]],
            }
            for name, vals in self.class_history.items()
        }
        payload = {
            "history": history,
            "class_history": class_history,
            "best_pck": float(self.best_pck),
            "best_epoch": int(self.best_epoch),
        }
        with open(hist_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _plot_curves(self):
        """Generate and save loss/PCK/Error curves."""
        epochs = self.history["epoch"]
        if len(epochs) == 0:
            return
        
        # 1) Loss curves
        plt.figure(figsize=(6,4))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = self.checkpoint_dir / "loss_curves.png"
        plt.savefig(loss_path, dpi=120)
        plt.close()
        
        # 2) Mean PCK + per-class PCK
        plt.figure(figsize=(7,5))
        plt.plot(epochs, self.history["mean_pck"], label="Mean PCK", linewidth=2)
        for name in self.class_names:
            plt.plot(epochs, self.class_history[name]["PCK"], linestyle="--", alpha=0.6, label=f"{name}_PCK")
        plt.xlabel("Epoch")
        plt.ylabel("PCK")
        plt.ylim(0, 1.0)
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pck_path = self.checkpoint_dir / "pck_curves.png"
        plt.savefig(pck_path, dpi=120)
        plt.close()
        
        # 3) Mean Error + per-class MRE
        plt.figure(figsize=(7,5))
        plt.plot(epochs, self.history["mean_error"], label="Mean Error", linewidth=2)
        for name in self.class_names:
            plt.plot(epochs, self.class_history[name]["MRE"], linestyle="--", alpha=0.6, label=f"{name}_MRE")
        plt.xlabel("Epoch")
        plt.ylabel("Error (px)")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        err_path = self.checkpoint_dir / "error_curves.png"
        plt.savefig(err_path, dpi=120)
        plt.close()
        
        # Optional: log to W&B
        if self.config.get("use_wandb", False):
            wandb.log({
                "plots/loss_curves": wandb.Image(str(loss_path)),
                "plots/pck_curves": wandb.Image(str(pck_path)),
                "plots/error_curves": wandb.Image(str(err_path)),
            }, step=len(epochs)-1)
