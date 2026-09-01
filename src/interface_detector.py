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

    The detector identifies objects, computes a center for cropping either as
    the plain center-of-mass of the segmentation mask, or (more robustly) as
    an intensity-weighted centroid restricted to the mask -- i.e. it only
    looks at pixels the mask says belong to the object, and within that
    region it up-weights the bright ("white", front-facing) pixels and
    down-weights the dark ("black", edge-on) pixels. This makes the crop
    center track the true visual front of the object even when the mask
    itself is skewed at shallow/edge-on viewing angles.

    It then crops a `crop_size` region around that center and zeros out
    background pixels (mask_mode=False), or just returns the mask
    (mask_mode=True).
    """

    def __init__(self, weights_path, device='cuda', conf_threshold=0.1, half=True,
                 mask_mode=True, crop_size=(512, 512),
                 intensity_weighted_center=True, brightness_gamma=1.0,
                 brightness_percentile_floor=0.0):
        """
        Initialize the InterfaceDetector with a YOLO model.

        Args:
            weights_path (str or Path): Path to YOLO weights file (.pt)
            device (str): Device to run inference on ('cuda' or 'cpu')
            conf_threshold (float): Confidence threshold for detections
            half (bool): whether to load the model in half precision or not for speed
            mask_mode (bool): do you want the original image sizes but with masked
                background (True) or cropped images according to crop_size (False)
            crop_size (tuple): what size to crop the images to if mask_mode is False
            intensity_weighted_center (bool): if True, compute the crop center as a
                brightness-weighted centroid restricted to the predicted mask
                (robust to edge-on angles). If False, fall back to the plain
                unweighted mask center-of-mass (your original behavior).
            brightness_gamma (float): exponent applied to the (0..1 normalized)
                brightness weights before averaging. >1 sharpens the weighting
                toward the brightest pixels (i.e. more aggressively pulls the
                center toward the whitest region); 1.0 = linear weighting.
            brightness_percentile_floor (float): fraction in [0, 1). Pixels inside
                the mask whose brightness falls below this percentile (computed
                over mask pixels only) get zero weight. Use e.g. 0.5 to only let
                the brightest half of the masked pixels vote on the center.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.half = half
        self.mask_mode = mask_mode
        self.crop_size = crop_size
        self.intensity_weighted_center = intensity_weighted_center
        self.brightness_gamma = brightness_gamma
        self.brightness_percentile_floor = brightness_percentile_floor

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

    @staticmethod
    def _to_luminance(image_chw):
        """
        Convert a [C, H, W] image tensor to a single-channel brightness/luminance
        map [H, W], robust to both grayscale and RGB input and to [0,1] or
        [0,255] value ranges (range doesn't matter since we only use relative
        weights).

        Args:
            image_chw (torch.Tensor): [C, H, W] image, C in {1, 3, 4}
        Returns:
            torch.Tensor: [H, W] float luminance map
        """
        img = image_chw.float()
        c = img.shape[0]
        if c == 1:
            lum = img[0]
        elif c >= 3:
            # standard luma weighting (Rec. 601), ignore alpha if present
            r, g, b = img[0], img[1], img[2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            lum = img.mean(dim=0)
        return lum

    def _intensity_weighted_center(self, image_chw, mask_hw):
        """
        Compute a brightness-weighted centroid, restricted strictly to the
        predicted object mask. Background pixels (mask == 0) are excluded
        entirely from the weighting, regardless of how bright/dark they are,
        so a bright background can never bias the result.

        Args:
            image_chw (torch.Tensor): [C, H, W] image (same frame the mask was
                predicted on)
            mask_hw (torch.Tensor): [H, W] binary mask (0/1 or bool)

        Returns:
            (int, int): (center_y, center_x) in pixel coordinates
        """
        mask_bool = mask_hw.to(torch.bool)
        y_coords, x_coords = torch.where(mask_bool)

        # Plain mask center-of-mass as a safe fallback
        fallback_y = torch.mean(y_coords.float())
        fallback_x = torch.mean(x_coords.float())

        if not self.intensity_weighted_center:
            return int(fallback_y.item()), int(fallback_x.item())

        lum = self._to_luminance(image_chw).to(mask_bool.device)
        masked_vals = lum[mask_bool]  # brightness values, mask pixels only

        if masked_vals.numel() == 0:
            return int(fallback_y.item()), int(fallback_x.item())

        # Normalize brightness within the mask to [0, 1] so weighting is
        # scale-invariant (works whether inputs are 0-1 or 0-255, and
        # regardless of overall exposure).
        v_min = masked_vals.min()
        v_max = masked_vals.max()
        denom = (v_max - v_min)
        if denom <= 1e-6:
            # Degenerate case: mask region has (almost) uniform brightness,
            # no gradient to exploit -> fall back to plain centroid.
            return int(fallback_y.item()), int(fallback_x.item())

        weights = (masked_vals - v_min) / denom

        # Optional hard floor: zero out the dimmer fraction of masked pixels
        # so only the brighter part of the face votes on the center.
        if self.brightness_percentile_floor > 0:
            thresh = torch.quantile(weights, self.brightness_percentile_floor)
            weights = torch.where(weights >= thresh, weights, torch.zeros_like(weights))

        # Optional gamma sharpening toward the brightest pixels
        if self.brightness_gamma != 1.0:
            weights = weights.clamp(min=0) ** self.brightness_gamma

        weight_sum = weights.sum()
        if weight_sum <= 1e-6:
            return int(fallback_y.item()), int(fallback_x.item())

        com_y = (y_coords.float() * weights).sum() / weight_sum
        com_x = (x_coords.float() * weights).sum() / weight_sum

        return int(com_y.item()), int(com_x.item())

    def forward(self, image, debug=False):
        """
        Process image(s) through YOLO and extract cropped, masked objects.

        Args:
            image (torch.Tensor): Input image(s)
                  Expected shape [B, C, H, W] or [C, H, W] with values in [0, 1] or [0, 255]

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

            # YOLO masks are often output at a different resolution than the
            # source image (e.g. 640x640 vs original HxW). Resize the mask to
            # match the image before doing pixel-wise brightness lookups so
            # coordinates line up correctly.
            if mask.shape[-2:] != (h, w):
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode='nearest'
                ).squeeze(0).squeeze(0).to(torch.uint8)
            else:
                mask_resized = mask

            com_y, com_x = self._intensity_weighted_center(
                image[batch_idx], mask_resized
            )
            batch_centers.append((com_y, com_x))

            if self.mask_mode:
                batch_masks.append(mask)
                continue
            # Calculate crop boundaries centered at the (intensity-weighted) center
            half_size = self.crop_size // 2

            # Determine crop region with bounds checking
            crop_y1 = max(0, com_y - half_size)
            crop_y2 = min(h, com_y + half_size)
            crop_x1 = max(0, com_x - half_size)
            crop_x2 = min(w, com_x + half_size)

            # Extract crop from image and mask
            cropped_img = image[batch_idx, :, crop_y1:crop_y2, crop_x1:crop_x2]  # [C, H_crop, W_crop]
            cropped_mask = mask_resized[crop_y1:crop_y2, crop_x1:crop_x2]  # [H_crop, W_crop]

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
            cropped_img = cropped_img[:, :self.crop_size, :self.crop_size].to(self.device)
            cropped_mask = cropped_mask[:self.crop_size, :self.crop_size].to(self.device)

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

    def __call__(self, image, debug=False):
        """Allow calling the detector directly"""
        return self.forward(image, debug)