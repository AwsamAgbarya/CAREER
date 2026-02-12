import sys
import os
from pathlib import Path
import argparse
import torch
from torchinfo import summary

# CUDA optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_CHECKPOINT_USE_REENTRANT'] = '0'

# VMamba paths
VMAMBA_ROOT = Path('./VMamba').resolve()
sys.path.insert(0, str(VMAMBA_ROOT))
sys.path.insert(0, str(VMAMBA_ROOT / 'segmentation'))

from model import MM_VSSM
from configs.vmamba_tiny_config import model_config as vmamba_config
from utils.dataset import KeypointDataset
from utils.trainer import SegmentationTrainer
from utils.model import VmambaSegmentor
from utils.metrics import compute_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train VMamba Keypoint Segmentation')
    
    # Data
    parser.add_argument('--data-dir', default='./renders/train/', help='Training data directory')
    parser.add_argument('--bg-dir', default='./backgrounds/', help='Background images directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/finetuned/vmamba2/', help='Checkpoint save directory')
    parser.add_argument('--weights-path', default='./renders/class_weights.pt', help='Class weights cache path')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--mode', default='segment_six', choices=['all', 'segment_six', 'segment_eight'])
    parser.add_argument('--crop-size', type=int, default=256, help='Input image size')
    parser.add_argument('--class-names', type=list, default=['background','hole_1', 'hole_2', 'hole_3', 'hole_4', 'hole_5', 'center'], help='Name of each keypoint class')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    
    # Optimization
    parser.add_argument('--backbone-lr', type=float, default=5e-5, help='Backbone learning rate')
    parser.add_argument('--decoder-lr', type=float, default=5e-4, help='Decoder learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--start-lr-factor', type=float, default=0.1, help='Warmup start LR factor')
    
    # Loss
    parser.add_argument('--ce-weight', type=float, default=0.0, help='Cross-entropy loss weight')
    parser.add_argument('--dice-weight', type=float, default=1.0, help='Dice loss weight')
    parser.add_argument('--focal-weight', type=float, default=1.0, help='Focal loss weight')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--boundary-weight', type=float, default=0.01, help='Focal loss gamma')
    
    # Regularization
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--no-ema', action='store_true', help='Disable EMA')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable gradient checkpointing')
    
    # Logging
    parser.add_argument('--vis-freq', type=int, default=5, help='Visualization frequency (epochs)')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', default='vmamba-keypoint-seg', help='W&B project name')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup dataset
    dataset = KeypointDataset(
        args.data_dir,
        background_directory=args.bg_dir,
        mode=args.mode,
        crop_size=(args.crop_size, args.crop_size)
    )
    print(f"Dataset: {len(dataset)} samples, {args.num_classes} classes, mode={args.mode}")
    
    # Load or compute class weights
    weights_path = Path(args.weights_path)
    if weights_path.exists():
        class_weights = torch.load(weights_path)
        print(f"Loaded class weights: {class_weights.numpy()}")
    else:
        class_weights = compute_class_weights(dataset, args.num_classes, args.num_workers)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(class_weights, weights_path)
        print(f"Computed class weights: {class_weights.numpy()}")
    
    # Setup model
    backbone = MM_VSSM(**vmamba_config).cuda()
    model = VmambaSegmentor(backbone, args.num_classes).cuda()
    
    # Enable gradient checkpointing
    if not args.no_checkpoint:
        if hasattr(model.backbone, 'set_grad_checkpointing'):
            model.backbone.set_grad_checkpointing(enable=True, use_reentrant=False)
        elif hasattr(model.backbone, 'gradient_checkpointing_enable'):
            model.backbone.gradient_checkpointing_enable()
        else:
            for module in model.backbone.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
        print("Gradient checkpointing enabled")
    
    summary(model, input_size=(1, 3, args.crop_size, args.crop_size))
    
    # Training config
    config = {
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'val_split': args.val_split,
        'backbone_lr': args.backbone_lr,
        'decoder_lr': args.decoder_lr,
        'start_lr': args.start_lr_factor,
        'weight_decay': args.weight_decay,
        'ce_weight': args.ce_weight,
        'dice_weight': args.dice_weight,
        'focal_weight': args.focal_weight,
        'focal_gamma': args.focal_gamma,
        'smoothness_weight': args.boundary_weight,
        'ema_decay': args.ema_decay,
        'use_ema': not args.no_ema,
        'checkpoint_dir': args.checkpoint_dir,
        'vis_frequency': args.vis_freq,
        'use_wandb': args.wandb,
        'wandb_project': args.wandb_project,
        'num_classes': args.num_classes,
        'class_names': args.class_names
    }
    
    # Train
    trainer = SegmentationTrainer(model, dataset, class_weights, config, device='cuda')
    trainer.train()
    
    print(f"\nTraining complete! Best checkpoint: {args.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    main()