import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class MultiClassEdgeAwareModule(nn.Module):
    """Multi-class edge detection for complex urban scenarios.

    Generates per-class edge attention instead of single-channel edge map.
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        self.num_classes = num_classes

        # Edge detection with larger receptive field
        self.edge_conv = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=5,  # Larger kernel for better edge detection
                padding=2,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels // 2,
                in_channels // 4,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)  # Multi-class output
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            edge_attention: [B, num_classes, H, W] - per-class edge attention
        """
        edge_attention = torch.sigmoid(self.edge_conv(x))
        return edge_attention


class EnhancedDetailRefinementModule(nn.Module):
    """Enhanced detail refinement with larger receptive field."""

    def __init__(self, in_channels):
        super().__init__()

        # Multi-scale detail extraction with larger kernels
        self.detail_conv1 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.detail_conv2 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=7,  # Larger kernel for better detail capture
            padding=3,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # Fusion
        self.fusion = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            refined: [B, C, H, W]
        """
        detail1 = self.detail_conv1(x)
        detail2 = self.detail_conv2(x)

        detail_features = torch.cat([detail1, detail2], dim=1)
        refined = self.fusion(detail_features)

        return refined


class OptimizedHighResolutionEnhancement(nn.Module):
    """Optimized HRDE for multi-class urban scene segmentation.

    Key improvements:
    1. Multi-class edge attention (per-class instead of single-channel)
    2. Larger receptive field in detail refinement (7x7 vs 5x5)
    3. Smaller initial fusion weight (0.05 vs 0.3) for gentler enhancement
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        self.edge_module = MultiClassEdgeAwareModule(in_channels, num_classes)
        self.detail_module = EnhancedDetailRefinementModule(in_channels)

        # Adaptive fusion weight - much smaller initial value for UAVid
        self.fusion_weight = nn.Parameter(torch.tensor(0.05))

        # Per-class weighting for edge attention
        self.class_weights = nn.Parameter(torch.ones(num_classes))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - fused features from FPN
        Returns:
            enhanced: [B, C, H, W] - enhanced features
        """
        # Extract multi-class edge attention
        edge_attention = self.edge_module(x)  # [B, num_classes, H, W]

        # Apply learnable per-class weights
        class_weights = torch.softmax(self.class_weights, dim=0)  # Normalize
        edge_attention = edge_attention * class_weights.view(1, -1, 1, 1)

        # Aggregate to single-channel attention
        edge_attention_agg = edge_attention.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Refine details
        detail_features = self.detail_module(x)

        # Apply edge-aware enhancement with gentle fusion
        fusion_weight = torch.sigmoid(self.fusion_weight)
        enhanced = x + fusion_weight * edge_attention_agg * detail_features

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_optimized(UPerHead):
    """UPerHead with Optimized HRDE for UAVid.

    Optimized for multi-class complex urban scenarios with:
    - Multi-class edge detection
    - Larger receptive field
    - Gentler fusion (weight=0.05)
    """

    def __init__(self, num_classes=8, **kwargs):
        # Pass num_classes to parent
        super().__init__(num_classes=num_classes, **kwargs)

        # Optimized HRDE module
        self.hr_enhancement = OptimizedHighResolutionEnhancement(
            in_channels=self.channels,
            num_classes=num_classes
        )

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)

        # Apply bottleneck
        feats = self.fpn_bottleneck(fpn_outs)

        # Apply optimized HRDE enhancement
        feats = self.hr_enhancement(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
