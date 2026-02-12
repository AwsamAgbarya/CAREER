import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import kornia.morphology as morph
import kornia.augmentation as K
from tqdm import tqdm

class KeypointDataset(Dataset):
    def __init__(self, data_directory, crop_size=(128,128), background_directory=None, mode="segment_six", fix_holes=True):  # Generate 10 variants per image
        
        self.data_directory = data_directory
        self.N = len(os.listdir(data_directory))//4
        self.H, self.W = 720, 1280
        self.crop_size = crop_size
        self.fix_holes = fix_holes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device} for preprocessing...")

        # Setup backgrounds
        self.backgrounds = []
        if background_directory:
            bg_files = [str(f) for f in Path(background_directory).rglob('*')
                       if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            print(f"Pre-loading {len(bg_files)} backgrounds...")
            for bg_file in tqdm(bg_files):
                bg = Image.open(bg_file).convert("RGB").resize((self.W, self.H), Image.Resampling.BILINEAR)
                self.backgrounds.append((pil_to_tensor(bg).float() / 255.0).to(device))
        
        # Setup class remapping
        if mode == "segment_six":
            self.accepted_classes = torch.tensor([0,3,5,7,8,10,12], dtype=torch.long).to(device)
            self.map_index = torch.tensor([0,1,2,3,4,5,6], dtype=torch.long).to(device)
        else:
            raise ValueError("Unavailable mode")
        
        self.class_remap_table = torch.zeros(256, dtype=torch.long).to(device)
        self.class_remap_table[self.accepted_classes] = self.map_index
        
        if self.fix_holes:
            self.morph_kernel = torch.ones(3, 3).to(device)

        self.images = torch.zeros((self.N, 3, crop_size[0], crop_size[1]), dtype=torch.float32)
        self.masks = torch.zeros((self.N, crop_size[0], crop_size[1]), dtype=torch.long)
        
        for idx_orig in tqdm(range(self.N), desc="GPU preprocessing"):
            prefix = os.path.join(self.data_directory, f"view_{idx_orig:05d}")
            img = pil_to_tensor(Image.open(f"{prefix}_rgb.jpeg").convert("RGB")).float().to(device) / 255.0
            mask = pil_to_tensor(Image.open(f"{prefix}_mask.png")).to(device)
            
            # Process single image
            img_aug = img
            mask_aug = mask
            
            # Background
            if len(self.backgrounds) > 0:
                bg = self.backgrounds[np.random.randint(len(self.backgrounds))]
                mask_3d = (mask_aug == 0).expand(3, -1, -1)
                img_aug = torch.where(mask_3d, bg, img_aug)
            
            # Remap + fill
            mask_aug = self.class_remap_table[mask_aug.long()]
            if self.fix_holes:
                mask_aug = self._fill_mask_holes(mask_aug)
            
            # Crop
            img_crop, mask_crop = crop_centered_on_mask(img_aug, mask_aug, crop_size=self.crop_size)
            
            # Store to CPU RAM
            self.images[idx_orig] = img_crop.cpu()
            self.masks[idx_orig] = mask_crop.squeeze(0).cpu()
        self.backgrounds = [bg.cpu() for bg in self.backgrounds]
        
        print(f"Dataset ready: {self.N} samples using {self.images.element_size() * self.images.nelement() / 1e9:.2f} GB RAM")
    
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]
    
    def _fill_mask_holes(self, mask):
        """Vectorized morphological closing"""
        unique_ids = torch.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        if len(unique_ids) == 0:
            return mask
        
        binary_masks = torch.stack([
            (mask == hole_id).float() for hole_id in unique_ids
        ])
        
        kernel = self.morph_kernel.to(mask.device)
        filled_batch = morph.closing(binary_masks, kernel)
        
        mask_fixed = torch.zeros_like(mask)
        for i, hole_id in enumerate(unique_ids):
            mask_fixed[filled_batch[i] > 0.5] = hole_id
        
        return mask_fixed
        
    def __len__(self):
        return self.N

def filter_and_remap_classes(mask, accepted_classes, map_indices):
    original_shape = mask.shape
    mask = mask.long()
    accepted_classes = accepted_classes.long()
    map_indices = map_indices.long()

    max_class = max(mask.max().item(), accepted_classes.max().item()) + 1
    remap_table = torch.zeros(max_class, dtype=torch.long)
    remap_table[accepted_classes] = map_indices
    
    output_mask = remap_table[mask.flatten()].reshape(original_shape)
    
    return output_mask

def crop_centered_on_mask(image, mask, crop_size=(512, 512)):
    """
    Crops image and mask centered on the centroid of non-zero mask pixels.
    
    Args:
        image: (C, H, W) or (H, W) tensor
        mask: (1, H, W) or (H, W) tensor with >0 for object pixels
        crop_size: (height, width) of output crop
    
    Returns:
        cropped_image, cropped_mask both of size crop_size
    """
    H, W = image.shape[-2:]
    target_h, target_w = crop_size
    
    # Find non-zero mask pixels
    mask_2d = mask.squeeze() if mask.ndim == 3 else mask
    y_coords, x_coords = torch.where(mask_2d > 0)
    
    if len(y_coords) == 0:
        # Fallback to image center if mask is empty
        center_y, center_x = H // 2, W // 2
    else:
        # Compute centroid (center of mass)
        center_y = int(torch.mean(y_coords.float()).item())
        center_x = int(torch.mean(x_coords.float()).item())
    
    # Calculate crop boundaries
    start_y = center_y - (target_h // 2)
    start_x = center_x - (target_w // 2)
    
    # Clamp to image boundaries
    start_y = max(0, min(start_y, H - target_h))
    start_x = max(0, min(start_x, W - target_w))
    
    end_y = start_y + target_h
    end_x = start_x + target_w
    
    # Crop
    cropped_image = image[..., start_y:end_y, start_x:end_x]
    cropped_mask = mask[..., start_y:end_y, start_x:end_x]
    
    return cropped_image, cropped_mask


def show_tensor_image(tensor):
    img = tensor.clone().detach().cpu()
    img = img.permute(1, 2, 0)
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()