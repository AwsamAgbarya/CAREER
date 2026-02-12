import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import torch.nn.functional as F
from time import time

class InterfaceDetector:
    """
    YOLO-based detector that extracts and crops objects from images.
    
    The detector identifies objects, computes the center of mass of the mask,
    crops a 512x512 region around it, and zeros out background pixels.
    """
    
    def __init__(self, weights_path, device='cuda', conf_threshold=0.1, half=True, mask_mode=True, crop_size=(512,512)):
        """
        Initialize the InterfaceDetector with a YOLO model.
        
        Args:
            weights_path (str or Path): Path to YOLO weights file (.pt)
            device (str): Device to run inference on ('cuda' or 'cpu')
            conf_threshold (float): Confidence threshold for detections
            half (bool): whether to load the model in half precision or not for speed
            mask_mode (bool): do you want the original image sizes but with masked background (True) or cropped images according to crop_size (False)
            crop_size (tuple): what size to crop the images to if mask_mode is fasle
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.half = half
        self.mask_mode = mask_mode
        self.crop_Size = crop_size
        # Load YOLO model
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading YOLO model from {weights_path}...")
        self.model = YOLO(str(weights_path))
        self.model.to(self.device)

        if self.half and self.device == 'cuda':
            self.model.model.half()  # Convert model weights to FP16
            print(f"Model loaded on {self.device} with FP16")
        else:
            print(f"Model loaded on {self.device}")
    
    def forward(self, image, crop_size=512, debug=False):
        """
        Process image(s) through YOLO and extract cropped, masked objects.
        
        Args:
            image (torch.Tensor): Input image(s)
                  Expected shape [B, C, H, W] or [C, H, W] with values in [0, 1] or [0, 255]
            crop_size (int): Size of the output square crop (default: 512)
        
        Returns:
            torch.Tensor: Cropped and masked images [B, C, crop_size, crop_size]
                         Background pixels are set to [0, 0, 0]
            list: List of (com_y, com_x) tuples for each image in batch
                  Returns None for images where no object is detected
        """
        if debug:
            start_time = time()
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)
            single_image = True
        else:
            single_image = False
            
        b, c, h, w = image.shape
        
        # Run YOLO inference on batch
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            agnostic_nms=True,
            max_det=1,
            half=self.half
        )
        
        # Process each image in batch
        batch_outputs = []
        batch_centers = []
        batch_masks = []
        
        for batch_idx in range(b):
            result = results[batch_idx]
            if len(result.boxes) == 0:
                return None, None
            
            # Get mask for this image
            mask = result.masks[0].data[0]
            mask = (mask > 0.5).to(torch.uint8)
            
            # Compute center of mass of the mask
            if mask.sum() == 0:
                return None, None
            
            y_coords, x_coords = torch.where(mask > 0)
            com_y = int(torch.mean(y_coords.float()).item())
            com_x = int(torch.mean(x_coords.float()).item())
            batch_centers.append((com_y, com_x))

            if self.mask_mode:
                batch_masks.append(mask)
                continue
            # Calculate crop boundaries centered at center of mass
            half_size = crop_size // 2
            
            # Determine crop region with bounds checking
            crop_y1 = max(0, com_y - half_size)
            crop_y2 = min(h, com_y + half_size)
            crop_x1 = max(0, com_x - half_size)
            crop_x2 = min(w, com_x + half_size)
            
            # Extract crop from image and mask
            cropped_img = image[batch_idx, :, crop_y1:crop_y2, crop_x1:crop_x2]  # [C, H_crop, W_crop]
            cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]  # [H_crop, W_crop]
            
            # Pad if necessary to reach crop_size x crop_size
            pad_top = max(0, half_size - com_y)
            pad_bottom = max(0, (com_y + half_size) - h)
            pad_left = max(0, half_size - com_x)
            pad_right = max(0, (com_x + half_size) - w)
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                # Pad image [C, H, W] -> padding format is (left, right, top, bottom)
                cropped_img = F.pad(
                    cropped_img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=0
                )
                # Pad mask [H, W]
                cropped_mask = F.pad(
                    cropped_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=0
                ).squeeze(0).squeeze(0)  # [H, W]
            
            # Ensure exact size
            cropped_img = cropped_img[:, :crop_size, :crop_size].to(self.device)
            cropped_mask = cropped_mask[:crop_size, :crop_size].to(self.device)
            
            # Zero out background pixels (where mask == 0)
            cropped_mask_3ch = cropped_mask.unsqueeze(0)  # [1, H, W] - broadcast to channels
            masked_output = cropped_img * cropped_mask_3ch  # [C, H, W]
            
            batch_outputs.append(masked_output)
        
        # Stack valid outputs
        if self.mask_mode:
            valid_outputs = batch_masks 
        else:
            valid_outputs = batch_outputs 
        output_tensor = torch.stack(valid_outputs, dim=0)  # [B, C, crop_size, crop_size]
        
        if debug:
            end_time = time()
            print(f"Cropping {b} images took {(end_time-start_time)*1000:3f}ms")
        # Return single image if input was single
        if single_image:
            return output_tensor.squeeze(0), batch_centers[0]
        
        return output_tensor, batch_centers
    
    def __call__(self, image, crop_size=512, debug=False):
        """Allow calling the detector directly"""
        return self.forward(image, crop_size,debug)