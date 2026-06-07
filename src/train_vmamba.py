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
from utils.trainer import KeypointTrainer
from utils.model import VmambaSegmentor

'''
python ./src/train_vmamba.py
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Train VMamba Keypoint Segmentation')
    
    # Data
    parser.add_argument('--data-dir', default='./renders/train/', help='Training data directory')
    parser.add_argument('--bg-dir', default='./backgrounds', help='Background images directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/finetuned/vmamba_heat_compound3/', help='Checkpoint save directory')
    parser.add_argument('--weights-path', default='./renders/class_weights.pt', help='Class weights cache path')
    parser.add_argument('--sigma', type=float, default=6.0)
    # Model
    parser.add_argument('--num-classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--mode', default='segment_all', choices=['segment_all', 'segment_six', 'segment_eight'])
    parser.add_argument('--crop-size', type=int, default=256, help='Input image size')
    parser.add_argument('--class-names', type=list, default=['hole_1', 'hole_2', 'hole_3', 'hole_4', 'hole_5', 'hole6', 'hole7', 'hole8', 'hole9', 'hole10', 'center'], help='Name of each keypoint class')
    parser.add_argument('--H', type=int, default=720)
    parser.add_argument('--W', type=int, default=1280)

    # Training
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    
    # Optimization
    parser.add_argument('--backbone-lr', type=float, default=3e-5, help='Backbone learning rate')
    parser.add_argument('--decoder-lr', type=float, default=3e-3, help='Decoder learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--start-lr-factor', type=float, default=0.1, help='Warmup start LR factor')
    
    # Loss
    parser.add_argument('--alpha', type=float, default=2.1)
    parser.add_argument('--omega', type=float, default=14.0)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--w_weight', type=float, default=1.0)
    parser.add_argument('--mse_weight', type=float, default=0.1)
    parser.add_argument('--cross_weight', type=float, default=0.05)
    parser.add_argument('--sharp_weight', type=float, default=0.05)
    parser.add_argument('--distance_factor', type=float, default=2.5)
    parser.add_argument('--metric_threshold', type=float, default=10.0)
    
    # Regularization
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--no-ema', action='store_true', help='Disable EMA')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable gradient checkpointing')
    
    # Logging
    parser.add_argument('--vis-freq', type=int, default=10, help='Visualization frequency (epochs)')
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
        crop_size=(args.crop_size, args.crop_size),
        sigma=args.sigma,
        H=args.H,
        W=args.W,
        max_N=1750
    )
    print(f"Dataset: {len(dataset)} samples, {args.num_classes} classes, mode={args.mode}")
    
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
        'sigma': args.sigma,
        'alpha': args.alpha,
        'omega': args.omega,
        'epsilon': args.epsilon,
        'theta': args.theta,
        'w_weight': args.w_weight,
        'mse_weight': args.mse_weight,
        'cross_weight': args.cross_weight,
        'sharp_weight': args.sharp_weight,
        'distance_factor': args.distance_factor,
        'metric_threshold': args.metric_threshold,
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
    trainer = KeypointTrainer(model, dataset, config, device='cuda')
    trainer.train()
    
    print(f"\nTraining complete! Best checkpoint: {args.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    main()