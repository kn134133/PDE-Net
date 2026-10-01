import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class EdgeAwareModule(nn.Module):
    """Extract and enhance edge features for better boundary segmentation."""

    def __init__(self, in_channels):
        super().__init__()

        # Edge detection branch
        self.edge_conv = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
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
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            edge_attention: [B, 1, H, W] - edge attention map
        """
        edge_attention = torch.sigmoid(self.edge_conv(x))
        return edge_attention


class DetailRefinementModule(nn.Module):
    """Refine features with focus on small objects and fine details."""

    def __init__(self, in_channels):
        super().__init__()

        # Multi-scale detail extraction
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
            kernel_size=5,
            padding=2,
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


class HighResolutionEnhancement(nn.Module):
    """Enhance features for high-resolution processing.

    Combines edge-aware attention and detail refinement to improve
    segmentation of small objects and object boundaries.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.edge_module = EdgeAwareModule(in_channels)
        self.detail_module = DetailRefinementModule(in_channels)

        # Adaptive fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - fused features from FPN
        Returns:
            enhanced: [B, C, H, W] - enhanced features
        """
        # Extract edge attention
        edge_attention = self.edge_module(x)

        # Refine details
        detail_features = self.detail_module(x)

        # Apply edge-aware enhancement
        # Areas with high edge attention get more detail refinement
        fusion_weight = torch.sigmoid(self.fusion_weight)
        enhanced = x + fusion_weight * edge_attention * detail_features

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_v2(UPerHead):
    """UPerHead with High-Resolution Detail Enhancement.

    Based on standard UPerHead (not UPerHead_lwc), compatible with
    standard UPerHead checkpoints. Adds HRDE module after bottleneck.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # High-resolution enhancement module
        self.hr_enhancement = HighResolutionEnhancement(
            in_channels=self.channels
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

        # Apply HRDE enhancement
        feats = self.hr_enhancement(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
