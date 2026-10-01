import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class LightweightDetailEnhancer(nn.Module):
    """Lightweight detail enhancement for a single FPN level.

    Applied RIGHT AFTER spattn and BEFORE FPN fusion to preserve
    fine-grained details at each scale.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Very lightweight: single 3x3 conv for detail extraction
        self.detail_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,  # Depthwise conv for efficiency
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # 1x1 conv to mix channels
        self.channel_mix = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None  # No activation
        )

        # Learnable fusion weight, initialized very small
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - features at one FPN level
        Returns:
            enhanced: [B, C, H, W]
        """
        # Extract details with depthwise conv
        details = self.detail_conv(x)
        details = self.channel_mix(details)

        # Residual enhancement with learned weight
        weight = torch.sigmoid(self.fusion_weight)
        enhanced = x + weight * details

        return enhanced


@MODELS.register_module()
class UPerHead_MSDE(UPerHead):
    """UPerHead with Multi-Scale Detail Enhancement.

    Key insight from SP framework analysis:
    - SP fuses global info at each stage output via spattn
    - FPN then fuses multi-scale features
    - Details get lost during these fusions

    Solution: Enhance details at EACH FPN level BEFORE fusion
    - Lightweight enhancers preserve details at each scale
    - Applied after lateral convs, before top-down fusion
    - Much earlier than traditional HRDE (which acts after bottleneck)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Detail enhancers for each FPN level
        # Apply to lateral features (after lateral_convs, before fusion)
        self.detail_enhancers = nn.ModuleList([
            LightweightDetailEnhancer(self.channels)
            for _ in range(len(self.in_channels))
        ])

    def forward(self, inputs):
        """Forward function with multi-scale detail enhancement."""

        inputs = self._transform_inputs(inputs)

        # Build laterals with detail enhancement
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral = lateral_conv(inputs[i])
            # ENHANCE DETAILS HERE - before FPN fusion!
            lateral = self.detail_enhancers[i](lateral)
            laterals.append(lateral)

        # PSP module on last level
        laterals.append(self.psp_forward(inputs))

        # Build top-down path (standard FPN)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # Build outputs (standard FPN)
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

        # Final classification
        output = self.cls_seg(feats)

        return output
