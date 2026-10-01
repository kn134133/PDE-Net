import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head_lwc import UPerHead_lwc


class ScalePredictor(nn.Module):
    """Predicts scale category (large/medium/small) for each pixel."""

    def __init__(self, in_channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales

        # Lightweight predictor
        self.conv1 = ConvModule(
            in_channels,
            in_channels // 4,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.conv2 = nn.Conv2d(in_channels // 4, num_scales, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: feature map [B, C, H, W]
        Returns:
            scale_probs: [B, num_scales, H, W] - probability for each scale
        """
        x = self.conv1(x)
        scale_logits = self.conv2(x)
        scale_probs = F.softmax(scale_logits, dim=1)
        return scale_probs


class AdaptiveAggregation(nn.Module):
    """Adaptively aggregate multi-scale features based on scale prediction.

    Instead of concatenating features, this module computes a weighted sum
    where weights depend on the predicted scale at each spatial location.
    """

    def __init__(self, num_scales=3, num_levels=4):
        super().__init__()
        self.num_scales = num_scales
        self.num_levels = num_levels

        # Learnable weights for each scale-level combination
        # Shape: [num_scales, num_levels]
        # scale 0 (large): prefer high-level features (C5, C4)
        # scale 1 (medium): balanced
        # scale 2 (small): prefer low-level features (C2, C3)
        init_weights = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],  # large: more weight on C5 (last)
            [0.25, 0.25, 0.25, 0.25],  # medium: balanced
            [0.4, 0.3, 0.2, 0.1],  # small: more weight on C2 (first)
        ], dtype=torch.float32)

        self.scale_level_weights = nn.Parameter(init_weights)

    def forward(self, features, scale_probs):
        """
        Args:
            features: list of [B, C, H, W], len=num_levels (all same spatial size)
            scale_probs: [B, num_scales, H, W]
        Returns:
            aggregated: [B, C, H, W] - single feature map (not concatenated)
        """
        B, C, H, W = features[0].shape

        # Normalize weights across levels for each scale
        weights = F.softmax(self.scale_level_weights, dim=1)  # [num_scales, num_levels]

        # Compute weighted features for each scale
        aggregated = 0
        for scale_idx in range(self.num_scales):
            scale_weight = scale_probs[:, scale_idx:scale_idx+1, :, :]  # [B, 1, H, W]

            # Aggregate features for this scale
            scale_features = 0
            for level_idx in range(self.num_levels):
                level_weight = weights[scale_idx, level_idx]
                scale_features = scale_features + level_weight * features[level_idx]

            # Weight by scale probability
            aggregated = aggregated + scale_weight * scale_features

        return aggregated


@MODELS.register_module()
class UPerHead_SAFA(UPerHead_lwc):
    """UPerHead with Scale-Aware Feature Aggregation.

    This head adaptively aggregates multi-scale features based on predicted
    object scales, giving more weight to high-level features for large objects
    and low-level features for small objects.
    """

    def __init__(self, num_scales=3, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = num_scales

        # Scale predictor
        self.scale_predictor = ScalePredictor(
            in_channels=self.channels,
            num_scales=num_scales
        )

        # Adaptive aggregation
        self.adaptive_aggregation = AdaptiveAggregation(
            num_scales=num_scales,
            num_levels=len(self.in_channels)
        )

        # Replace conv_fusion to match SAFA output channels
        # Original expects: len(self.in_channels) * self.channels
        # SAFA outputs: self.channels
        # So we create a new conv_fusion that takes self.channels as input
        from .uper_head_lwc import conv3x3_bn_relu
        self.conv_fusion_safa = conv3x3_bn_relu(self.channels, self.channels, 1)

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
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        # 3. Upsample all features to same size
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        # 4. SAFA: Scale-aware aggregation instead of simple concatenation
        # Original: fusion_out = torch.cat(fusion_list, 1)

        # Get mean feature for scale prediction
        fpn_out_mean = torch.stack(fusion_list, dim=0).mean(dim=0)

        # Predict scale for each pixel
        scale_probs = self.scale_predictor(fpn_out_mean)  # [B, num_scales, H, W]

        # Adaptively aggregate features
        safa_out = self.adaptive_aggregation(fusion_list, scale_probs)

        # 5. Final fusion and classification
        # Use SAFA-specific conv_fusion
        x = self.conv_fusion_safa(safa_out)
        out = self.object_head(x)

        return F.interpolate(out, size=seg_size, mode='bilinear', align_corners=False)
