import torch
import torchvision.transforms.functional as TF
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from skimage.morphology import remove_small_holes
from ultralytics import YOLO
import yaml
import random
import json
'''
python ./train_yolo.py --rotate 180 --img_size 640
python ./train_yolo.py --skip_dataset_prep --dataset_yaml ./yolov8_dataset/data.yaml --batch_size 8
'''

def parse_args():
    """Parse command line arguments for YOLOv8 finetuning pipeline."""
    parser = argparse.ArgumentParser(
        description='Finetune YOLOv8 segmentation model on custom dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Dataset preparation arguments
    dataset_group = parser.add_argument_group('Dataset Preparation')
    dataset_group.add_argument('--renders_dir',type=str,default='./renders/train',help='Directory containing rendered images and masks')
    dataset_group.add_argument('--backgrounds_dir',type=str,default='./backgrounds/',help='Directory containing the backgrounds')
    dataset_group.add_argument('--dataset-output_dir',type=str,default='./yolov8_dataset',help='Output directory for prepared YOLO dataset')
    dataset_group.add_argument('--train_ratio',type=float,default=0.8,help='Ratio of training data (rest will be validation)')
    dataset_group.add_argument('--hole_size',type=int,default=50,help='Maximum hole size to fill in masks (in pixels)')
    dataset_group.add_argument('--translate',type=tuple,default=(0.5, 0.5),help='Maximum translation across x,y image coordinates, be careful the maximum value should be 0.5, as translating the object from the center to the edge is 50% of the image size')
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--pretrained_model',type=str,default='./checkpoints/pretrained/yolov8s-seg.pt',help='Path to pretrained YOLOv8 model weights')
    model_group.add_argument('--finetuned_output',type=str,default='./checkpoints/finetuned',help='Output directory for finetuned model checkpoints')
    model_group.add_argument('--cache_dir',type=str,default='./checkpoints/finetuned/cache_dir/',help='Output directory for cache files downloaded by Ultralytics')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs',type=int,default=300,help='Number of training epochs')
    train_group.add_argument('--batch_size',type=int,default=8 ,help='Training batch size')
    train_group.add_argument('--img_size',type=int,default=640,help='Input image size for training')
    train_group.add_argument('--lr',type=float,default=0.01,help='starting lr rate for training')
    train_group.add_argument('--device',type=int,default=0,help='GPU device ID (use -1 for CPU)')
    
    # Additional options
    parser.add_argument('--skip_dataset_prep',action='store_true',help='Skip dataset preparation step (use existing dataset)')
    parser.add_argument('--dataset_yaml',type=str,default=None,help='Path to existing dataset YAML (if skipping preparation)')
    
    args = parser.parse_args()
    
    # Validation
    if args.skip_dataset_prep and args.dataset_yaml is None:
        parser.error("--dataset-yaml must be provided when using --skip-dataset-prep")
    
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")
    
    return args

def fill_mask_holes(mask, hole_size=50):
    """
    Fill small holes in the mask to create clean polygon boundaries.
    
    Args:
        mask: Binary mask image (0=background, 1/2+=foreground)
        hole_size: Maximum hole size to fill in pixels
    
    Returns:
        Filled binary mask
    """
    # Convert to binary (anything > 0 is foreground)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Fill small holes using morphological operations
    filled_mask = remove_small_holes(binary_mask.astype(bool), area_threshold=hole_size)
    
    return filled_mask.astype(np.uint8)


def mask_to_polygon_coords(mask, img_height, img_width):
    """
    Convert a binary mask to normalized polygon coordinates (YOLOv8 segmentation format).
    
    Args:
        mask: Binary mask image
        img_height: Height of the original image
        img_width: Width of the original image
    
    Returns:
        List of normalized polygon coordinates [x1, y1, x2, y2, ...] in range [0, 1]
        Returns None if mask is empty
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to reduce number of points
    # 0.02 * perimeter gives a good balance between accuracy and simplicity
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    # Convert contour points to normalized coordinates
    polygon_coords = []
    for point in approx:
        x, y = point[0]
        # Normalize to [0, 1]
        norm_x = x / img_width
        norm_y = y / img_height
        # Clamp to [0, 1] in case of floating point errors
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        polygon_coords.append(norm_x)
        polygon_coords.append(norm_y)
    
    return polygon_coords if polygon_coords else None


def prepare_dataset(renders_dir, output_dir, backgrounds_dir, train_ratio=0.8, hole_size=50, translate=(0.5, 0.5)):
    """
    Prepare dataset from render images and masks for YOLOv8 segmentation training.
    
    Args:
        renders_dir: Path to directory containing view_*_rgb.png and view_*_mask.png
        output_dir: Path where to save train/val split
        backgrounds_dir: Path to directory containing background images
        train_ratio: Ratio for train/val split (default 0.8)
        hole_size: Maximum hole size to fill in masks
        rotate: maximum rotation angle
        translate: maximum translation across the screen xy coordinates
    """
    renders_path = Path(renders_dir)
    output_path = Path(output_dir)
    backgrounds_path = Path(backgrounds_dir)

    # Load all background images recursively from all subdirectories
    background_files = list(backgrounds_path.rglob('*.png')) + \
                    list(backgrounds_path.rglob('*.jpg')) + \
                    list(backgrounds_path.rglob('*.jpeg'))

    if not background_files:
        raise ValueError(f"No background images found in {backgrounds_dir} or its subdirectories")

    print(f"Found {len(background_files)} background images across all subdirectories")
    
    # Create output directories
    train_img_dir = output_path / 'train' / 'images'
    train_lbl_dir = output_path / 'train' / 'labels'
    val_img_dir = output_path / 'val' / 'images'
    val_lbl_dir = output_path / 'val' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all image-mask pairs
    rgb_files = sorted(renders_path.glob('view_*_rgb.jpeg'))
    print(f"Found {len(rgb_files)} image-mask pairs")
    
    # Extract indices and create pairs
    pairs = []
    for rgb_file in rgb_files:
        stem = rgb_file.stem  # e.g., 'view_00000_rgb'
        index = stem.split('_')[1]  # Extract index
        mask_file = renders_path / f'view_{index}_mask.png'
        
        if mask_file.exists():
            pairs.append((rgb_file, mask_file))
    
    print(f"Found {len(pairs)} valid image-mask pairs")
    
    # Split into train and val
    train_pairs, val_pairs = train_test_split(pairs, train_size=train_ratio, random_state=42)
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    processed_count = 0
    for split, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
        valid_samples = 0
        split_img_dir = train_img_dir if split == 'train' else val_img_dir
        split_lbl_dir = train_lbl_dir if split == 'train' else val_lbl_dir
        for rgb_file, mask_file in split_pairs:
            # Read image and mask
            img = cv2.imread(str(rgb_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"Warning: Could not read {rgb_file} or {mask_file}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Random translation parameters (sample once, apply to both)
            trans_x = np.random.uniform(-translate[0] * img_width / 2, translate[0] * img_width / 2)
            trans_y = np.random.uniform(-translate[1] * img_height / 2, translate[1] * img_height / 2)
            affine_matrix = np.float32([
                [1, 0, trans_x],
                [0, 1, trans_y]
            ])
            
            # Apply the SAME transformation to both image and mask
            img = cv2.warpAffine(img, affine_matrix, (img_width, img_height), 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = cv2.warpAffine(mask, affine_matrix, (img_width, img_height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Randomly select a background image
            bg_file = np.random.choice(background_files)
            background = cv2.imread(str(bg_file))
            
            if background is None:
                print(f"Warning: Could not read background {bg_file}")
                continue
            
            # Resize background to match image dimensions
            background = cv2.resize(background, (img_width, img_height))
            
            # Create composite image: background where mask==0, foreground where mask>0
            # Create 3-channel mask for broadcasting
            mask_3ch = np.stack([mask > 0] * 3, axis=-1)
            # Composite: keep object pixels (mask > 0), replace background pixels (mask == 0)
            composite_img = np.where(mask_3ch, img, background)
            
            # Fill holes in mask for polygon extraction
            filled_mask = fill_mask_holes(mask, hole_size=hole_size)
            
            # Convert mask to polygon coordinates
            polygon_coords = mask_to_polygon_coords(filled_mask, img_height, img_width)
            
            if polygon_coords is None:
                print(f"Warning: No valid polygon found in {mask_file}")
                continue
            
            # FIX: Use the same base filename for both image and label
            # Keep the original filename (e.g., "view_00000_rgb")
            base_name = rgb_file.stem  # "view_00000_rgb"
            
            # Save composite image with original name
            output_img_path = split_img_dir / f"{base_name}.jpeg"
            cv2.imwrite(str(output_img_path), composite_img)
            
            # Save label file with matching base name
            output_lbl_path = split_lbl_dir / f"{base_name}.txt"
            
            with open(output_lbl_path, 'w') as f:
                # Format: class_id x1 y1 x2 y2 ... xn yn
                # Class 0 for single object
                f.write(f"0 {' '.join(map(str, polygon_coords))}\n")
            valid_samples += 1
            processed_count += 1
        print(f"Created {valid_samples} valid labels for {split} set")
    print(f"Successfully processed {processed_count} images")
        
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['object']
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Dataset prepared at {output_path}")
    print(f"Dataset config saved to {yaml_path}")
    return str(yaml_path)

def finetune_model(
    dataset_yaml,
    model_path='../checkpoints/pretrained/yolov8s-seg.pt',
    output_dir='../checkpoints/finetuned',
    cache_dir='../checkpoints/finetuned/cache',
    epochs=100,
    batch_size=16,
    img_size=1024,
    lr0=0.01,
    device=0
):
    """
    Finetune YOLOv8 nano segmentation model on custom dataset.
    """
    # Create base output directory
    os.environ['ULTRALYTICS_CACHE'] = cache_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    current_seed = random.randint(0, 2**32 - 1)
    print(f"Generated Training Seed: {current_seed}")

    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Starting finetuning...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=25,
        save=True,
        single_cls=True,
        project=str(output_path),
        name='yolov8s-seg-finetuned',
        exist_ok=False,
        verbose=True,
        lr0=lr0,
        seed=current_seed,
        deterministic=True
    )
    
    # Retrieve the ACTUAL save directory used
    actual_save_dir = Path(model.trainer.save_dir)
    
    # Save the seed into that specific directory
    seed_file = actual_save_dir / 'training_seed.json'
    with open(seed_file, 'w') as f:
        json.dump({'seed': current_seed}, f, indent=4)
        
    print(f"Seed saved to {seed_file}")
    best_weight_path = actual_save_dir / 'weights' / 'best.pt'
    return str(best_weight_path)


if __name__ == '__main__':
    args = parse_args()
    
    # Step 1: Prepare dataset
    if not args.skip_dataset_prep:
        print("=" * 60)
        print("Step 1: Preparing dataset...")
        print("=" * 60)
        dataset_yaml = prepare_dataset(
            renders_dir=args.renders_dir,
            output_dir=args.dataset_output_dir,
            backgrounds_dir=args.backgrounds_dir,
            train_ratio=args.train_ratio,
            hole_size=args.hole_size,
            translate=args.translate
        )
    else:
        print("Skipping dataset preparation, using existing dataset")
        dataset_yaml = args.dataset_yaml
    
    # Step 2: Finetune model
    print("\n" + "=" * 60)
    print("Step 2: Finetuning YOLOv8 nano model...")
    print("=" * 60)
    finetuned_model_path = finetune_model(
        dataset_yaml=dataset_yaml,
        model_path=args.pretrained_model,
        output_dir=args.finetuned_output,
        cache_dir=args.cache_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr0=args.lr,
        device=args.device
    )

    print("\n" + "=" * 60)
    print("Finetuning complete!")
    print(f"Finetuned model saved to: {finetuned_model_path}")
    print("=" * 60)