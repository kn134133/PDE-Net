import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class ScaleSpecificDetailExtractor(nn.Module):
    """Extract details at specific scale using dilated convolutions.

    Different object types need different receptive fields:
    - Large objects (buildings, roads): small dilation
    - Medium objects (trees, vegetation): medium dilation
    - Small objects (cars, humans): large dilation to capture context
    """

    def __init__(self, in_channels, dilation=1):
        super().__init__()

        # Depthwise separable conv for efficiency
        self.depthwise = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.pointwise = ConvModule(
            in_channels,
            in_channels // 4,  # Changed from //3 to //4 for divisibility
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GeometricRegularityModule(nn.Module):
    """Enhance geometric regularity for structured objects (buildings, roads).

    UAVid is aerial imagery - buildings have strong horizontal/vertical edges.
    Use directional filters to capture this.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Horizontal edge detector
        self.h_conv = ConvModule(
            in_channels,
            in_channels // 4,
            kernel_size=(1, 5),
            padding=(0, 2),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # Vertical edge detector
        self.v_conv = ConvModule(
            in_channels,
            in_channels // 4,
            kernel_size=(5, 1),
            padding=(2, 0),
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # Diagonal edges
        self.d_conv = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.fusion = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        d_feat = self.d_conv(x)

        feat = torch.cat([h_feat, v_feat, d_feat], dim=1)
        feat = self.fusion(feat)
        return feat


class ScaleAwareDetailEnhancement(nn.Module):
    """Scale-Aware Detail Enhancement for UAVid.

    Key insights for UAVid:
    1. Multi-scale: buildings (large), trees (medium), cars/humans (small)
    2. Geometric regularity: buildings have straight edges (H/V directions)
    3. Context matters: small objects need larger receptive field
    4. Conservative fusion: start with very small weight
    """

    def __init__(self, in_channels):
        super().__init__()

        # Multi-scale detail extraction with different dilations
        self.small_scale = ScaleSpecificDetailExtractor(in_channels, dilation=1)   # Large objects
        self.medium_scale = ScaleSpecificDetailExtractor(in_channels, dilation=2)  # Medium objects
        self.large_scale = ScaleSpecificDetailExtractor(in_channels, dilation=4)   # Small objects + context

        # Geometric regularity for structured objects
        self.geometric = GeometricRegularityModule(in_channels)

        # Adaptive fusion with scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Final fusion (3*C//4 + C = 7C//4)
        fusion_in_channels = (in_channels // 4) * 3 + in_channels
        self.fusion = ConvModule(
            fusion_in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # Very conservative initial weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced: [B, C, H, W]
        """
        # Extract multi-scale details
        small = self.small_scale(x)   # [B, C//3, H, W]
        medium = self.medium_scale(x)  # [B, C//3, H, W]
        large = self.large_scale(x)    # [B, C//3, H, W]

        # Compute scale attention
        scale_attn = self.scale_attention(x)  # [B, 3, 1, 1]

        # Concatenate all scales (each is C//4, total 3*C//4)
        multi_scale = torch.cat([small, medium, large], dim=1)  # [B, 3C//4, H, W]

        # Extract geometric features
        geometric = self.geometric(x)  # [B, C, H, W]

        # Fuse multi-scale and geometric
        combined = torch.cat([multi_scale, geometric], dim=1)  # [B, 3C//4 + C = 7C//4, H, W]
        detail_features = self.fusion(combined)  # [B, C, H, W]

        # Conservative residual enhancement
        weight = torch.sigmoid(self.fusion_weight)
        enhanced = x + weight * detail_features

        return enhanced


@MODELS.register_module()
class UPerHead_SADE(UPerHead):
    """UPerHead with Scale-Aware Detail Enhancement.

    Specifically designed for UAVid characteristics:
    - Multi-scale objects (buildings to humans)
    - Geometric regularity in aerial view
    - Conservative enhancement to avoid overfitting
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Scale-aware detail enhancement
        self.sade = ScaleAwareDetailEnhancement(self.channels)

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

        # Apply scale-aware detail enhancement
        feats = self.sade(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
