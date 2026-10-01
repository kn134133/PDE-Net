import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class HRDE_V4(nn.Module):
    """HRDE V4: Fixed version addressing fundamental issues.

    Key improvements over V3:
    1. NO BOTTLENECK: Keep 512 channels throughout (not 512→256→1)
    2. SEMANTIC boundaries: Use feature similarity, not visual edges
    3. CLASS-SPECIFIC strength: Learn per-class enhancement weights

    This avoids:
    - Information loss (V3's bottleneck)
    - Texture confusion (V3 enhanced grass texture as boundaries)
    - Damaging strong classes (V3 hurt Low veg -0.96%)
    """

    def __init__(self, in_channels=512, num_classes=8):
        super().__init__()

        # 1. Semantic boundary detector (NO BOTTLENECK!)
        # Keep ALL channels to preserve semantic information
        self.boundary_conv1 = ConvModule(
            in_channels,
            in_channels,  # Keep 512!
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.boundary_conv2 = ConvModule(
            in_channels,
            in_channels,  # Keep 512!
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )

        # 2. Detail refinement (NO BOTTLENECK!)
        self.detail_conv1 = ConvModule(
            in_channels,
            in_channels,  # Keep 512!
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.detail_conv2 = ConvModule(
            in_channels,
            in_channels,  # Keep 512!
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.detail_conv3 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None
        )

        # 3. Class-specific enhancement strength
        # Initialize based on baseline performance:
        # Building (93%) → 0.1, Road (80%) → 0.2, Low veg (71%) → 0.3
        init_alphas = torch.tensor([
            0.1,  # Building - already strong
            0.2,  # Road - medium
            0.2,  # Tree - medium
            0.3,  # Low vegetation - weak, needs enhancement
            0.25, # Background - weak
            0.2,  # Moving car - medium
            0.2,  # Static car - medium
            0.15, # Human - tiny, don't over-enhance
        ])
        self.class_alphas = nn.Parameter(init_alphas)

    def compute_semantic_boundary(self, boundary_feats):
        """Detect SEMANTIC boundaries using feature similarity.

        Key insight: Grass has texture edges everywhere, but we only want
        to enhance where grass MEETS road (semantic boundary), not
        internal grass texture.

        Method: Neighboring pixels with similar features → same class → no boundary
        """
        B, C, H, W = boundary_feats.shape

        # Normalize features for similarity computation
        boundary_feats_norm = F.normalize(boundary_feats, p=2, dim=1)

        # Compute feature similarity with neighbors (horizontal)
        feats_shifted_h = F.pad(boundary_feats_norm[:, :, :, 1:], (0, 1), value=0)
        similarity_h = torch.sum(
            boundary_feats_norm * feats_shifted_h, dim=1, keepdim=True
        )  # [B, 1, H, W], range [-1, 1]

        # Compute feature similarity with neighbors (vertical)
        feats_shifted_v = F.pad(boundary_feats_norm[:, :, 1:, :], (0, 0, 0, 1), value=0)
        similarity_v = torch.sum(
            boundary_feats_norm * feats_shifted_v, dim=1, keepdim=True
        )

        # Average similarity
        similarity = (similarity_h + similarity_v) / 2  # [B, 1, H, W]

        # Convert similarity to boundary
        # High similarity (close to 1) → same class → no boundary (close to 0)
        # Low similarity (close to -1) → different class → strong boundary (close to 1)
        semantic_boundary = (1 - similarity) / 2  # Normalize to [0, 1]

        # Apply sigmoid to sharpen boundaries
        semantic_boundary = torch.sigmoid(semantic_boundary * 10 - 5)

        return semantic_boundary

    def forward(self, x, seg_logits=None):
        """
        Args:
            x: [B, 512, H, W] - feature maps
            seg_logits: [B, num_classes, H, W] - segmentation logits for class-specific weighting
        Returns:
            enhanced: [B, 512, H, W] - enhanced features
        """
        B, C, H, W = x.shape

        # 1. Detect semantic boundaries (keep 512 channels!)
        boundary_feats = self.boundary_conv1(x)
        boundary_feats = self.boundary_conv2(boundary_feats)
        semantic_boundary = self.compute_semantic_boundary(boundary_feats)  # [B, 1, H, W]

        # 2. Refine details (keep 512 channels!)
        detail_feats = self.detail_conv1(x)
        detail_feats = self.detail_conv2(detail_feats)
        detail_feats = self.detail_conv3(detail_feats)  # [B, 512, H, W]

        # 3. Class-specific enhancement strength
        if seg_logits is not None and self.training:
            # Use predicted class probabilities to weight enhancement per-pixel
            class_probs = F.softmax(seg_logits.detach(), dim=1)  # [B, num_classes, H, W]

            # Resize if needed
            if class_probs.shape[2:] != (H, W):
                class_probs = F.interpolate(
                    class_probs, size=(H, W), mode='bilinear', align_corners=False
                )

            # Weight by class-specific alphas
            alpha_map = torch.einsum('bchw,c->bhw', class_probs, self.class_alphas)
            alpha_map = alpha_map.unsqueeze(1)  # [B, 1, H, W]
        else:
            # Inference: use average alpha
            alpha_map = self.class_alphas.mean()

        # 4. Apply enhancement: x + alpha * boundary * detail
        # Only enhance at semantic boundaries, with class-specific strength
        enhanced = x + alpha_map * semantic_boundary * detail_feats

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_V4(UPerHead):
    """UPerHead with HRDE V4.

    HRDE V4 fixes fundamental issues:
    - No information bottleneck (keeps 512 channels)
    - Semantic boundaries (not texture edges)
    - Class-specific enhancement (don't hurt strong classes)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hrde = HRDE_V4(in_channels=self.channels, num_classes=kwargs.get('num_classes', 8))

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

        # Get initial segmentation logits for class-aware enhancement
        seg_logits = self.cls_seg(feats)

        # Apply HRDE V4 with class-aware enhancement
        feats_enhanced = self.hrde(feats, seg_logits=seg_logits)

        # Final classification on enhanced features
        output = self.cls_seg(feats_enhanced)

        return output
