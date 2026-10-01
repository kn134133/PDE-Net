import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class SmallObjectFeatureAmplifier(nn.Module):
    """Small Object Feature Amplification (SOFA).

    Target: Improve small object segmentation (Human 41%, Cars 68-70%)

    Strategy:
    1. Detect small object regions using gradient magnitude
       - High gradient + small area = likely small object
    2. Amplify features in those regions
    3. Keep large object regions unchanged
    """

    def __init__(self, in_channels):
        super().__init__()

        # Gradient detector for finding boundaries (where small objects are)
        self.grad_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.grad_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # Initialize as Sobel filters
        with torch.no_grad():
            # Sobel X
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            self.grad_x.weight.copy_(sobel_x)

            # Sobel Y
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            self.grad_y.weight.copy_(sobel_y)

        # Freeze gradient filters
        self.grad_x.weight.requires_grad = False
        self.grad_y.weight.requires_grad = False

        # Feature enhancement for small objects
        self.small_obj_enhancer = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=None  # No activation, will use as residual
            )
        )

        # Amplification factor (learnable, but start conservative)
        self.amplify_factor = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced: [B, C, H, W] - features with small objects amplified
        """
        # Detect boundaries using gradient magnitude
        grad_x = self.grad_x(x)  # [B, C, H, W]
        grad_y = self.grad_y(x)  # [B, C, H, W]

        # Gradient magnitude (L2 norm across channels)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # [B, C, H, W]
        grad_mag = grad_mag.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normalize gradient to [0, 1]
        grad_mag_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        # High gradient regions are likely object boundaries
        # Apply max pooling to suppress isolated noise
        boundary_mask = F.max_pool2d(grad_mag_norm, kernel_size=3, stride=1, padding=1)

        # Threshold: only enhance regions with high gradient (potential small objects)
        # Top 30% gradient regions
        threshold = torch.quantile(boundary_mask.flatten(1), 0.7, dim=1, keepdim=True).view(-1, 1, 1, 1)
        small_obj_mask = (boundary_mask > threshold).float()  # [B, 1, H, W]

        # Enhance features for small objects
        enhanced_features = self.small_obj_enhancer(x)  # [B, C, H, W]

        # Amplify small object features
        amplify = torch.sigmoid(self.amplify_factor)  # Bound to [0, 1], then scale
        amplification = 1.0 + amplify * 2.0  # Range [1.0, 3.0]

        # Apply mask: amplify small object regions, keep others unchanged
        output = x + small_obj_mask * amplification * enhanced_features

        return output


@MODELS.register_module()
class UPerHead_SOFA(UPerHead):
    """UPerHead with Small Object Feature Amplification.

    Specifically targets UAVid's weakest classes:
    - Human: 41.42% IoU
    - Moving car: 70.92% IoU
    - Static car: 68.35% IoU

    Strategy: Detect and amplify features in high-gradient regions
    where small objects are likely located.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Small object feature amplifier
        self.sofa = SmallObjectFeatureAmplifier(in_channels=self.channels)

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # Build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])

        # Upsample and concatenate
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)

        # Apply bottleneck
        feats = self.fpn_bottleneck(fpn_outs)

        # Apply small object feature amplification
        feats = self.sofa(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
