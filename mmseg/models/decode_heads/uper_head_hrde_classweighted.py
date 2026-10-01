import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class ClassWeightedDetailEnhancement(nn.Module):
    """Class-Weighted Detail Enhancement for UAVid.

    Key insight: Different object classes need different detail enhancement:
    - Large objects (buildings, roads): already perform well (93%, 80%), need LESS enhancement
    - Small objects (humans, cars): perform poorly (41%, 68%), need MORE enhancement

    UAVid 8 classes (index 0-7):
    0: Background clutter - 68.25% (medium, needs moderate enhancement)
    1: Building - 93.2% (large, needs minimal enhancement)
    2: Road - 80.74% (large, needs minimal enhancement)
    3: Tree - 80.4% (large, needs minimal enhancement)
    4: Low vegetation - 71.56% (medium, needs moderate enhancement)
    5: Moving car - 70.92% (small, needs HIGH enhancement)
    6: Static car - 68.35% (small, needs HIGH enhancement)
    7: Human - 41.42% (VERY small, needs MAXIMUM enhancement)
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        # Predict semantic pseudo-labels to identify which class needs enhancement
        self.class_predictor = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
            nn.Softmax(dim=1)  # [B, num_classes, H, W]
        )

        # Class-specific enhancement weights (learnable, but initialized based on performance)
        # Higher weight = more enhancement needed
        initial_weights = torch.tensor([
            0.5,  # 0: Background clutter (68.25%) - moderate
            0.1,  # 1: Building (93.2%) - minimal, already good
            0.2,  # 2: Road (80.74%) - low
            0.2,  # 3: Tree (80.4%) - low
            0.5,  # 4: Low vegetation (71.56%) - moderate
            0.8,  # 5: Moving car (70.92%) - high
            0.8,  # 6: Static car (68.35%) - high
            1.0,  # 7: Human (41.42%) - MAXIMUM, worst class!
        ])
        self.class_weights = nn.Parameter(initial_weights)

        # Multi-scale detail extraction for small objects
        # Small objects need larger receptive field
        self.detail_small_scale = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        self.detail_large_scale = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=5,
            padding=4,
            dilation=2,  # Larger receptive field for small objects
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

        # Global fusion weight (conservative)
        self.global_fusion_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - fused features from FPN
        Returns:
            enhanced: [B, C, H, W]
        """
        # Predict class probabilities for each pixel
        class_probs = self.class_predictor(x)  # [B, num_classes, H, W]

        # Compute class-weighted enhancement mask
        # Apply sigmoid to class_weights to bound them [0, 1]
        weights = torch.sigmoid(self.class_weights).view(1, -1, 1, 1)  # [1, num_classes, 1, 1]

        # Weighted sum: pixels predicted as "human" get weight 1.0, "building" get 0.1
        enhancement_mask = (class_probs * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Extract multi-scale details
        detail_small = self.detail_small_scale(x)
        detail_large = self.detail_large_scale(x)
        detail_features = self.fusion(torch.cat([detail_small, detail_large], dim=1))

        # Apply class-weighted enhancement
        global_weight = torch.sigmoid(self.global_fusion_weight)
        enhancement = global_weight * enhancement_mask * detail_features

        enhanced = x + enhancement

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_ClassWeighted(UPerHead):
    """UPerHead with Class-Weighted HRDE.

    Specifically targets UAVid's weakness: poor performance on small objects.
    - Human (41.42%) gets maximum enhancement
    - Cars (68-70%) get high enhancement
    - Buildings (93%) get minimal enhancement
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Class-weighted detail enhancement
        self.hrde = ClassWeightedDetailEnhancement(
            in_channels=self.channels,
            num_classes=8
        )

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

        # Apply class-weighted detail enhancement
        feats = self.hrde(feats)

        # Final classification
        output = self.cls_seg(feats)

        return output
