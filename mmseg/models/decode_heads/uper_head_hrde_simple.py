import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class DetailRefinementModule(nn.Module):
    """Refine features with focus on small objects and fine details.

    Simplified version without edge-aware module for multi-class scenarios.
    """

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


class SimplifiedDetailEnhancement(nn.Module):
    """Simplified detail enhancement for multi-class segmentation.

    Removes edge-aware module, uses only multi-scale detail refinement
    with smaller fusion weight to avoid disrupting pretrained features.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.detail_module = DetailRefinementModule(in_channels)

        # Smaller fusion weight for gentler enhancement
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - fused features from FPN
        Returns:
            enhanced: [B, C, H, W] - enhanced features
        """
        # Refine details
        detail_features = self.detail_module(x)

        # Gentle residual enhancement
        fusion_weight = torch.sigmoid(self.fusion_weight)
        enhanced = x + fusion_weight * detail_features

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_simple(UPerHead):
    """UPerHead with Simplified Detail Enhancement.

    Simplified version without edge-aware module, suitable for
    multi-class complex scenarios like UAVid.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Simplified detail enhancement module
        self.detail_enhancement = SimplifiedDetailEnhancement(
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

        # Apply simplified detail enhancement
        feats = self.detail_enhancement(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
