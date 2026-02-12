import os
import glob
import argparse
from time import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
from torch.nn.functional import pad
from kornia.filters import bilateral_blur

from interface_detector import InterfaceDetector

class AdaptiveKalmanFilter:
    def __init__(self, device='cuda', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.x = None
        self.P = None
        
        self.Q_base = torch.tensor(1e-5, device=device, dtype=dtype)
        self.R_base = torch.tensor(1e-4, device=device, dtype=dtype)
        self.Q_scale = torch.tensor(1e-6, device=device, dtype=dtype)
        self.R_scale = torch.tensor(5e-5, device=device, dtype=dtype)
    
    def get_adaptive_noise(self, depth):
        Q = self.Q_base + torch.abs(depth) * self.Q_scale
        R = self.R_base + torch.abs(depth) * self.R_scale
        return Q, R
    
    def filter(self, measurement):
        z = measurement.squeeze().to(device=self.device, dtype=self.dtype)
        
        if self.x is None:
            # Initialize state with first measurement
            self.x = z.clone()
            self.P = torch.ones_like(z) * 1e-3
            return self.x.unsqueeze(0)
        
        Q, R = self.get_adaptive_noise(self.x)
        self.P = self.P + Q
        
        K = self.P / (self.P + R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        
        return self.x.unsqueeze(0)
    
    def reset(self):
        self.x = None
        self.P = None


def parse_args():
    parser = argparse.ArgumentParser(description='Pose estimation pipeline')
    
    # Data
    parser.add_argument('--data-dir', help='Data image directory that contains "left" and "right" directories')
    parser.add_argument('--background-dir', help='Augmentation to add backgrounds to the images')
    parser.add_argument('--yolo-weights', default="./checkpoints/finetuned/yolov8s-seg-finetuned/weights/best.pt", help='Data image directory that contains finetuned yolo model weights')
    parser.add_argument('--process-depths', action='store_true', help='Whether to expect depth maps alongside the images or not')
    parser.add_argument('--crop-size', default=(256,256), help='Tuple of final image size if return-mask is not ON')

    return parser.parse_args()

def main():
    args = parse_args()
    left_folder = 'left'
    right_folder = 'right'
    process_depths = args.process_depths

    if not os.path.exists(args.data_dir):
        raise("Data directory does not exist")
    if (not os.path.exists(os.path.join(args.data_dir, left_folder))) or (not os.path.exists(os.path.join(args.data_dir, right_folder))):
        raise("Data directory does not contain left / right directories")
    
    output_dir = os.path.join(args.data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    left_images = sorted(glob.glob(args.data_dir + '/' + left_folder + '/' + '*_rgb.jpeg'))
    left_depths = sorted(glob.glob(args.data_dir + '/' + left_folder + '/' + '*_depth.png'))

    right_images = sorted(glob.glob(args.data_dir + '/' + right_folder + '/' + '*_rgb.jpeg'))
    right_depths = sorted(glob.glob(args.data_dir + '/' + right_folder + '/' + '*_depth.png'))

    backgrounds = []
    if args.background_dir:
        bg_files = [str(f) for f in Path(args.background_dir).rglob('*')
                    if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        print(f"Pre-loading {len(bg_files)} backgrounds...")
        for bg_file in tqdm(bg_files):
            bg = Image.open(bg_file).convert("RGB").resize((640, 360), Image.Resampling.BILINEAR)
            backgrounds.append(pil_to_tensor(bg).float() / 255.0)

    N = min([len(left_images) , len(right_images)])
    # half_h = args.crop_size[0] // 2
    # half_w = args.crop_size[1] // 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = InterfaceDetector(weights_path=args.yolo_weights, device=device, mask_mode=True, crop_size=args.crop_size)

    left_kalman = AdaptiveKalmanFilter(device=device)
    right_kalman = AdaptiveKalmanFilter(device=device)

    timer = []

    # Warmup the model
    dummy_tensor = torch.empty([3, 1280, 1024])
    detector(dummy_tensor)

    for i in range(400):
        left_img = pil_to_tensor(Image.open(left_images[i]).convert('RGB')).float()/255.0
        right_img = pil_to_tensor(Image.open(right_images[i]).convert('RGB')).float()/255.0

        if process_depths:
            left_dp = pil_to_tensor(Image.open(left_depths[i])).float()
            right_dp = pil_to_tensor(Image.open(right_depths[i])).float()

        if len(backgrounds) > 0:
            bg = backgrounds[np.random.randint(len(backgrounds))]

            mask_left = (left_dp == 0).expand(3, -1, -1)
            mask_right = (right_dp == 0).expand(3, -1, -1)

            left_img = torch.where(mask_left, bg, left_img)
            right_img = torch.where(mask_right, bg, right_img)

        start_time = time()
        input = torch.stack([left_img, right_img], dim=0).to(device)
        input_shape = input.shape
        if (input_shape[2]/32)%1!=0 or (input_shape[3]/32)%1!=0:
            padding = (0,int(((input_shape[3] // 32) + 1 - (input_shape[3]/32)) * 32)%32,
                        0, int(((input_shape[2] // 32) + 1 - (input_shape[2]/32)) * 32)%32)
            input = pad(input, padding, 'circular')

        H,W = input.shape[-2:]
        masks, centers = detector(input)
        if centers is None or centers[0] is None or centers[1] is None:
            print(f"Didnt find any objects in frame {i}")
            continue
        output = input * masks.unsqueeze(1)

        save_image(input[0], os.path.join(output_dir, f'input_image_{i}_left.jpeg'))
        save_image(input[1], os.path.join(output_dir, f'input_image_{i}_right.jpeg'))
        save_image(output[0], os.path.join(output_dir, f'output_image_{i}_left.jpeg'))
        save_image(output[1], os.path.join(output_dir, f'output_image_{i}_right.jpeg'))

        if process_depths:
            # left_dp = pil_to_tensor(Image.open(left_depths[i])).float()
            # right_dp = pil_to_tensor(Image.open(right_depths[i])).float()
            left_temporal = left_kalman.filter(left_dp)
            right_temporal = right_kalman.filter(right_dp)

            dps = torch.stack([left_temporal, right_temporal], dim=0).to(device)
            if (input_shape[2]/32)%1!=0 or (input_shape[3]/32)%1!=0:
                padding = (0,int(((input_shape[3] // 32) + 1 - (input_shape[3]/32)) * 32)%32,
                           0, int(((input_shape[2] // 32) + 1 - (input_shape[2]/32)) * 32)%32)
                dps = pad(dps, padding, 'circular')

            # mask = torch.zeros((2, 1, H, W), device=device)
            # for j, (cy, cx) in enumerate(centers):
            #     y_min = max(0, cy - half_h)
            #     y_max = min(H, cy + half_h)
            #     x_min = max(0, cx - half_w)
            #     x_max = min(W, cx + half_w)
                
            #     mask[j, 0, y_min:y_max, x_min:x_max] = 1
            # filtered = torch.where(masks==1, dps, 0.0)

            filtered = dps * masks.unsqueeze(1)

            save_image(filtered[0], os.path.join(output_dir, f'output_depth_{i}_left.jpeg'))
            save_image(filtered[1], os.path.join(output_dir, f'output_depth_{i}_right.jpeg'))

        timer.append(time()-start_time)
    print(f"Average timeframe processing time: {torch.tensor(timer).mean()*1000:3f}ms")
if __name__ == '__main__':
    main()