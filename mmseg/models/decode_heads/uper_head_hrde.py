import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head_lwc import UPerHead_lwc


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
    """Enhance features specifically for high-resolution (2048) processing.

    Combines edge-aware attention and detail refinement to improve
    segmentation of small water bodies and water body boundaries.
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
class UPerHead_HRDE(UPerHead_lwc):
    """UPerHead with High-Resolution Detail Enhancement.

    Designed to be trained from 1024 checkpoint on 2048 resolution data.
    Adds lightweight enhancement modules that focus on edges and small objects
    without disrupting the already-learned feature representations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # High-resolution enhancement module
        self.hr_enhancement = HighResolutionEnhancement(
            in_channels=self.channels
        )

    def forward(self, x):
        """Forward function following UPerHead_lwc architecture."""
        seg_size = x[-1].shape[-2:]

        # 1. PSP module on last feature
        conv5 = x[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))

        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        # 2. FPN top-down pathway
        fpn_feature_list = [f]
        for i in reversed(range(len(x) - 1)):
            conv_x = x[i]
            conv_x = self.fpn_in[i](conv_x)

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()

        # 3. Upsample all features to same size
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        # 4. Concatenate features
        fusion_out = torch.cat(fusion_list, 1)

        # 5. Apply HRDE enhancement
        fusion_out = self.conv_fusion(fusion_out)
        enhanced_features = self.hr_enhancement(fusion_out)

        # 6. Final classification
        out = self.object_head(enhanced_features)

        return F.interpolate(out, size=seg_size, mode='bilinear', align_corners=False)
