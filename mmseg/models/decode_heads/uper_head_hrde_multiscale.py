import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class MultiScaleHRDE(nn.Module):
    """Multi-Scale High-Resolution Detail Enhancement.

    Key insight for UAVid:
    - Large classes (Low veg 29.51%, Road 27.20%) need large receptive field
    - Medium classes (Background 13.14%) need medium receptive field
    - Small objects (Cars 2.51%) need small receptive field

    Previous HRDE failed because 3x3 conv is too small for large-class boundaries.
    Solution: Multi-scale enhancement with different receptive fields.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Multi-scale edge detection (different receptive fields)
        # Scale 1: Small receptive field (3x3) - for cars, small details
        self.edge_small = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 4,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

        # Scale 2: Medium receptive field (dilation=2) - for background
        self.edge_medium = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 4,
                kernel_size=3,
                padding=2,
                dilation=2,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

        # Scale 3: Large receptive field (dilation=4) - for low veg/road boundaries
        self.edge_large = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 4,
                kernel_size=3,
                padding=4,
                dilation=4,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

        # Fuse multi-scale edges
        self.edge_fusion = nn.Conv2d(3, 1, kernel_size=1)

        # Multi-scale detail refinement
        # Small scale
        self.detail_small = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            )
        )

        # Medium scale
        self.detail_medium = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=2,
                dilation=2,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            )
        )

        # Large scale
        self.detail_large = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=4,
                dilation=4,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            )
        )

        # Fuse multi-scale details
        self.detail_fusion = ConvModule(
            in_channels // 2 * 3,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None
        )

        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, gt_semantic_seg=None):
        """
        Args:
            x: [B, C, H, W] - feature maps
            gt_semantic_seg: [B, H_gt, W_gt] - ground truth (optional)
        Returns:
            enhanced: [B, C, H, W] - multi-scale enhanced features
        """
        B, C, H, W = x.shape

        # Multi-scale edge detection
        edge_small = torch.sigmoid(self.edge_small(x))    # [B, 1, H, W] - for cars
        edge_medium = torch.sigmoid(self.edge_medium(x))  # [B, 1, H, W] - for background
        edge_large = torch.sigmoid(self.edge_large(x))    # [B, 1, H, W] - for low veg/road

        # Fuse multi-scale edges
        edges = torch.cat([edge_small, edge_medium, edge_large], dim=1)  # [B, 3, H, W]
        edge_map = torch.sigmoid(self.edge_fusion(edges))  # [B, 1, H, W]

        # Multi-scale detail refinement
        detail_small = self.detail_small(x)    # [B, C//2, H, W]
        detail_medium = self.detail_medium(x)  # [B, C//2, H, W]
        detail_large = self.detail_large(x)    # [B, C//2, H, W]

        # Fuse multi-scale details
        details = torch.cat([detail_small, detail_medium, detail_large], dim=1)  # [B, 3C//2, H, W]
        detail_feats = self.detail_fusion(details)  # [B, C, H, W]

        # Apply enhancement
        alpha = torch.sigmoid(self.fusion_weight)
        enhanced = x + alpha * edge_map * detail_feats

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_MultiScale(UPerHead):
    """UPerHead with Multi-Scale HRDE.

    Addresses UAVid's multi-scale nature:
    - Large classes (Low veg, Road) need large receptive field
    - Medium classes (Background) need medium receptive field
    - Small objects (Cars) need small receptive field
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Multi-scale HRDE module
        self.hrde = MultiScaleHRDE(in_channels=self.channels)

    def forward(self, inputs, data_samples=None):
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

        # Apply multi-scale HRDE
        feats = self.hrde(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
