import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import resize


class BoundaryExtractor(nn.Module):
    """Extract boundary information using Sobel operator."""

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            boundary: (B, 1, H, W) boundary magnitude
        """
        # Average across channels first
        if x.size(1) > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        # Apply Sobel
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)

        # Compute magnitude
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return boundary


class BoundaryAttention(nn.Module):
    """Boundary-aware attention module."""

    def __init__(self, in_channels, boundary_channels=1):
        super().__init__()
        self.boundary_conv = ConvModule(
            boundary_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

        self.attention_conv = ConvModule(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Sigmoid')
        )

        self.refine_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, features, boundary):
        """
        Args:
            features: (B, C, H, W) input features
            boundary: (B, 1, H, W) boundary map
        Returns:
            enhanced_features: (B, C, H, W)
        """
        # Resize boundary to match features
        if boundary.shape[2:] != features.shape[2:]:
            boundary = F.interpolate(
                boundary, size=features.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Process boundary
        boundary_feat = self.boundary_conv(boundary)

        # Generate attention weights
        concat_feat = torch.cat([features, boundary_feat], dim=1)
        attention = self.attention_conv(concat_feat)

        # Apply attention
        attended_feat = features * attention

        # Refine
        enhanced_feat = self.refine_conv(attended_feat)

        return enhanced_feat + features  # Residual connection


@MODELS.register_module()
class UPerHead_BAFE(BaseDecodeHead):
    """UPerNet Head with Boundary-Aware Feature Enhancement.

    This head enhances features using boundary information extracted
    from the input, helping the model focus on boundary regions.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        # PSP Module
        self.psp_modules = nn.ModuleList()
        for pool_scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels[-1],
                        self.channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
            )

        self.psp_bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            self.lateral_convs.append(
                ConvModule(
                    in_channels,
                    self.channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # Boundary extraction and attention
        self.boundary_extractor = BoundaryExtractor()
        self.boundary_attention = BoundaryAttention(self.channels)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        for psp_module in self.psp_modules:
            psp_out = psp_module(x)
            psp_out = resize(
                psp_out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            psp_outs.append(psp_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.psp_bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        """Forward function."""
        # Build FPN
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # PSP on top feature
        laterals.append(self.psp_forward(inputs))

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # FPN outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(self.fpn_convs))
        ]
        fpn_outs.append(laterals[-1])

        # Fuse all scales
        for i in range(len(fpn_outs)):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

        fused_features = torch.cat(fpn_outs, dim=1)
        fused_features = self.fpn_bottleneck(fused_features)

        # Extract boundary from input image
        # Assume inputs[0] is the lowest level feature (closest to input)
        boundary = self.boundary_extractor(inputs[0])

        # Apply boundary-aware attention
        enhanced_features = self.boundary_attention(fused_features, boundary)

        # Final prediction
        output = self.cls_seg(enhanced_features)

        return output
