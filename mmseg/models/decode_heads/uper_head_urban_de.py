import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class SemanticAwareDetailModule(nn.Module):
    """Semantic-aware detail enhancement for urban scenes.

    Instead of generic edge detection, use semantic guidance to selectively
    enhance details where they matter (small objects, boundaries between
    specific classes).
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        # Semantic guidance: predict which areas need detail enhancement
        # Small objects (cars, clutter) need more enhancement than large (buildings)
        self.semantic_guide = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
            nn.Softmax(dim=1)  # Soft semantic prediction
        )

        # Class-specific enhancement weights (learnable)
        # Small objects like cars/clutter need more enhancement
        self.class_weights = nn.Parameter(torch.tensor([
            0.3,  # class 0: building (large, less detail needed)
            0.5,  # class 1: road (medium)
            0.3,  # class 2: tree (large)
            0.3,  # class 3: low vegetation
            0.8,  # class 4: moving car (SMALL, needs detail!)
            0.8,  # class 5: static car (SMALL, needs detail!)
            0.6,  # class 6: human (VERY SMALL)
            0.9,  # class 7: clutter (SMALL mixed objects)
        ]))

        # Multi-scale detail extraction
        self.detail_conv_small = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.detail_conv_large = ConvModule(
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
            detail_features: [B, C, H, W]
            enhancement_mask: [B, 1, H, W] - where to enhance
        """
        # Semantic guidance: where are the small objects?
        semantic_prob = self.semantic_guide(x)  # [B, num_classes, H, W]

        # Compute enhancement mask based on class importance
        class_weights = torch.sigmoid(self.class_weights).view(1, -1, 1, 1)
        enhancement_mask = (semantic_prob * class_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Extract multi-scale details
        detail_small = self.detail_conv_small(x)
        detail_large = self.detail_conv_large(x)
        detail_features = self.fusion(torch.cat([detail_small, detail_large], dim=1))

        return detail_features, enhancement_mask


class ContextualRefinementModule(nn.Module):
    """Refine details using contextual information.

    For urban scenes, context matters: a car on road vs car on building
    should be enhanced differently.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Context aggregation with dilated convolutions
        self.context_conv1 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=2,
            dilation=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.context_conv2 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=4,
            dilation=4,
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
        context1 = self.context_conv1(x)
        context2 = self.context_conv2(x)
        refined = self.fusion(torch.cat([context1, context2], dim=1))
        return refined


class UrbanDetailEnhancement(nn.Module):
    """Detail enhancement specifically designed for urban aerial scenes.

    Key differences from generic HRDE:
    1. Semantic-aware: knows which objects need detail enhancement
    2. Selective: only enhances small objects, not everything
    3. Contextual: considers surrounding context
    4. Conservative: smaller initial fusion weight to avoid over-enhancement
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        self.semantic_detail = SemanticAwareDetailModule(in_channels, num_classes)
        self.contextual_refine = ContextualRefinementModule(in_channels)

        # Very conservative fusion weight for urban scenes
        self.fusion_weight = nn.Parameter(torch.tensor(0.02))

        # Noise suppression: avoid enhancing high-frequency noise
        self.noise_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - fused features from FPN
        Returns:
            enhanced: [B, C, H, W] - selectively enhanced features
        """
        # Get semantic-aware details and enhancement mask
        detail_features, enhancement_mask = self.semantic_detail(x)

        # Apply contextual refinement
        refined_details = self.contextual_refine(detail_features)

        # Selective enhancement: only enhance where mask indicates
        # Use sigmoid gating to avoid over-enhancement
        fusion_weight = torch.sigmoid(self.fusion_weight)
        noise_gate = torch.sigmoid(self.noise_gate)

        # Final enhancement with noise suppression
        enhancement = fusion_weight * enhancement_mask * noise_gate * refined_details
        enhanced = x + enhancement

        return enhanced


@MODELS.register_module()
class UPerHead_UrbanDE(UPerHead):
    """UPerHead with Urban-specific Detail Enhancement.

    Designed for complex urban aerial scenes like UAVid, with:
    - Semantic-aware selective enhancement
    - Focus on small objects (cars, humans, clutter)
    - Contextual refinement
    - Conservative fusion to avoid noise amplification
    """

    def __init__(self, num_classes=8, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

        # Urban detail enhancement module
        self.urban_enhancement = UrbanDetailEnhancement(
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

        # Apply urban-specific detail enhancement
        feats = self.urban_enhancement(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
