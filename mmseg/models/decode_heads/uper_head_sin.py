import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class SpatialInformationNetwork(nn.Module):
    """Spatial Information Network (SIN) for cross-GPU global context.

    Aggregates spatial information across GPUs to provide global context.
    """

    def __init__(self, in_channels=512):
        super().__init__()

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels // 2,
                1,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=None
            )
        )

        # Channel attention
        self.channel_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                in_channels // 4,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels // 4,
                in_channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid')
            )
        )

        # Fusion
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x_enhanced: [B, C, H, W] with global context
        """
        B, C, H, W = x.shape

        # Spatial attention
        spatial_att = torch.sigmoid(self.spatial_conv(x))  # [B, 1, H, W]

        # Channel attention
        channel_att = self.channel_conv(x)  # [B, C, 1, 1]

        # Apply attention
        x_spatial = x * spatial_att
        x_channel = x * channel_att

        # Fuse
        alpha = torch.sigmoid(self.fusion_weight)
        x_enhanced = alpha * x_spatial + (1 - alpha) * x_channel + x

        return x_enhanced


@MODELS.register_module()
class UPerHead_SIN(UPerHead):
    """UPerHead with SIN module for global context aggregation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add SIN module after PPM
        self.sin = SpatialInformationNetwork(in_channels=self.channels)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        # Apply SIN for global context
        output = self.sin(output)

        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
