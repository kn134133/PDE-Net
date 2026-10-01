import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList()
        # 1x1 conv
        self.convs.append(ConvModule(
            in_channels, out_channels, 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        ))
        # dilated convs
        for dilation in dilations:
            self.convs.append(ConvModule(
                in_channels, out_channels, 3,
                padding=dilation, dilation=dilation,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ))
        # global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        # fusion
        self.fusion = ConvModule(
            out_channels * (len(dilations) + 2), out_channels, 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = []
        for conv in self.convs:
            feats.append(conv(x))
        # global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        feats.append(global_feat)
        # fusion
        out = torch.cat(feats, dim=1)
        out = self.fusion(out)
        return out


class LowLevelFeatureFusion(nn.Module):
    """Fuse low-level features for better small object detection"""
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.low_conv = ConvModule(
            low_channels, out_channels, 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.high_conv = ConvModule(
            high_channels, out_channels, 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.fusion = ConvModule(
            out_channels * 2, out_channels, 3, padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, low_feat, high_feat):
        size = low_feat.shape[2:]
        low = self.low_conv(low_feat)
        high = self.high_conv(high_feat)
        high = F.interpolate(high, size=size, mode='bilinear', align_corners=False)
        fused = torch.cat([low, high], dim=1)
        out = self.fusion(fused)
        return out


@MODELS.register_module()
class UPerHead_MSFF(UPerHead):
    """UPerHead with Multi-Scale Feature Fusion for better small object detection"""

    def __init__(self,
                 aspp_dilations=[6, 12, 18],
                 aspp_channels=256,
                 use_low_level_fusion=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_low_level_fusion = use_low_level_fusion

        # ASPP module on FPN output (after fusion)
        self.aspp = ASPPModule(
            self.channels,  # FPN output channels
            aspp_channels,
            dilations=aspp_dilations
        )

        # Low-level feature fusion (use stage 0 features - highest resolution)
        if use_low_level_fusion:
            self.low_level_fusion = LowLevelFeatureFusion(
                low_channels=self.in_channels[0],  # stage 0
                high_channels=aspp_channels,  # ASPP output
                out_channels=self.channels
            )

        # Enhanced final conv
        self.enhanced_conv = ConvModule(
            self.channels * (2 if use_low_level_fusion else 1),
            self.channels,
            3, padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs):
        """Forward function."""
        # inputs: list of features from backbone stages
        # [stage0, stage1, stage2, stage3]

        # Original UPerHead forward (PSP + FPN)
        x = self._transform_inputs(inputs)  # len=4

        # PSP module
        psp_outs = [x[-1]]
        psp_outs.extend(self.psp_modules(x[-1]))
        psp_outs = torch.cat(psp_outs, dim=1)
        x_psp = self.bottleneck(psp_outs)

        # FPN
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals[-1] = laterals[-1] + x_psp

        # top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # FPN outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals))
        ]

        # Fuse FPN outputs
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_out = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_out)

        # Apply ASPP on FPN output
        output = self.aspp(output)

        # Low-level feature fusion
        if self.use_low_level_fusion:
            low_level_feat = inputs[0]  # stage 0, highest resolution
            fused_low = self.low_level_fusion(low_level_feat, output)
            # Upsample output to match low-level feature size
            output_upsampled = F.interpolate(
                output,
                size=fused_low.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            # Concatenate
            output = torch.cat([output_upsampled, fused_low], dim=1)
            output = self.enhanced_conv(output)
            # Downsample back to original size
            output = F.interpolate(
                output,
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

        output = self.cls_seg(output)
        return output
