import sys
import contextlib
from pathlib import Path
import torch

VMAMBA_ROOT = Path('./VMamba').resolve()
sys.path.insert(0, str(VMAMBA_ROOT))
sys.path.insert(0, str(VMAMBA_ROOT / 'segmentation'))

from model import MM_VSSM
from configs.vmamba_tiny_config import model_config as vmamba_config
from utils.model import VmambaSegmentor


class KeypointPredictor:

    def __init__(self, weights, num_keypoints = 6, input_size = 256, device = 'cuda', warmup_passes = 1):
        self.device        = torch.device(device)
        self.input_size    = input_size
        self.num_keypoints = num_keypoints

        backbone   = MM_VSSM(**vmamba_config).to(self.device)
        self.model = VmambaSegmentor(backbone, num_keypoints).to(self.device)
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)
        # Trainer saves under 'model_state_dict'; fall back for raw state_dict saves
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        self._warmup(warmup_passes)

    def _warmup(self, n = 3):
        """Kick-start CUDA kernels with n random forward passes."""
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self._infer(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def _infer(self, tensor):
        """Forward pass → sigmoid-activated heatmaps (B, K, H, W)."""
        autocast_ctx = (
            torch.autocast(device_type='cuda')
            if self.device.type == 'cuda'
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = self.model(tensor)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def forward(self, image):
        """
        Run keypoint detection on a single image or batch.

        Returns
        -------
        coords   : np.ndarray  (K, 2) or (B, K, 2) — [x, y] pixel coords
        heatmaps : torch.Tensor (K, H, W) or (B, K, H, W) — sigmoid heatmaps on CPU
        """
        # Batch of tensors (B, 3, H, W) passed directly (your pipeline case)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        heatmaps = self._infer(image)
        return heatmaps.cpu()


    def __call__(self, image):
        return self.forward(image)
