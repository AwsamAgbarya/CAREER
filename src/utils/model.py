import torch
from torch import nn
import torch.nn.functional as F

class KeypointSegHead(nn.Module):
    """Lightweight segmentation head optimized for keypoint detection"""
    def __init__(self, in_channels_list=[96, 192, 384, 768], num_classes=6, feature_size=256):
        super().__init__()
        
        # Reduce all backbone features to same channel dimension
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, feature_size, 1),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        # Fusion refinement
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Decoder (upsample from 1/4 to full resolution)
        self.decoder = nn.Sequential(
            # 1/4 -> 1/2
            nn.Conv2d(feature_size, feature_size // 2, 3, padding=1),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 1/2 -> 1/1
            nn.Conv2d(feature_size // 2, feature_size // 4, 3, padding=1),
            nn.BatchNorm2d(feature_size // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Final segmentation
            nn.Conv2d(feature_size // 4, num_classes, 1)
        )
    
    def forward(self, features):
        # Assume features = [feat_1/4, feat_1/8, feat_1/16, feat_1/32]
        # Fuse at highest resolution (1/4)
        target_size = features[0].shape[2:]
        
        fused = sum(
            F.interpolate(
                lateral_conv(feat),
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            for feat, lateral_conv in zip(features, self.lateral_convs)
        )
        
        # Refine fused features
        fused = self.fusion_conv(fused)
        
        # Decode to full resolution
        output = self.decoder(fused)
        
        return output


class VmambaSegmentor(nn.Module):
    def __init__(self, backbone, num_classes=9):
        super().__init__()
        self.backbone = backbone
        self.seg_head = KeypointSegHead(
            in_channels_list=[96, 192, 384, 768],  # VMamba-Tiny channels
            num_classes=num_classes,
            feature_size=256  # Reduced from 512 for speed
        )
    
    def forward(self, x):
        features = self.backbone(x)  # Multi-scale features
        seg_logits = self.seg_head(features)  # Already at input resolution
        return seg_logits  # Shape: [B, num_classes, H, W]