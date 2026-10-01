import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .uper_head import UPerHead


class GTWeightedHRDE(nn.Module):
    """Ground Truth Weighted High-Resolution Detail Enhancement.

    Key insight: UAVid has severe class imbalance (Building 93% vs Human 28%).
    HRDE should focus on enhancing weak small-object classes (Human, Cars),
    not waste effort on already-strong large classes (Building, Road).

    Use GT masks during training to guide where HRDE should enhance:
    - Human (class 7): weight = 3.0 (weakest, 28-45% IoU)
    - Moving car (class 5): weight = 2.0 (weak, 67-76% IoU)
    - Static car (class 6): weight = 2.0 (weak, 64-75% IoU)
    - Others (Building, Road, etc.): weight = 0.1 (strong, >78% IoU)
    """

    def __init__(self, in_channels, num_classes=8):
        super().__init__()

        # UAVid class weights based on baseline performance
        # Building(0): 92.42%, Road(1): 78.32%, Tree(2): 78.48%,
        # Low veg(3): 68.34%, BG(4): 65.98%
        # Moving car(5): 66.98%, Static car(6): 64.19%, Human(7): 28.57%
        self.register_buffer('class_weights', torch.tensor([
            0.1,  # Building - already strong
            0.1,  # Road - already strong
            0.1,  # Tree - already strong
            0.5,  # Low vegetation - medium
            0.3,  # Background clutter - medium
            2.0,  # Moving car - weak, needs enhancement
            2.0,  # Static car - weak, needs enhancement
            3.0,  # Human - weakest, needs most enhancement
        ]))

        # Edge detection module (unchanged from HRDE v3)
        self.edge_module = nn.Sequential(
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
                in_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

        # Detail refinement module (unchanged from HRDE v3)
        self.detail_module = nn.Sequential(
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
                in_channels // 2,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels // 2,
                in_channels,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=None
            )
        )

        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))

    def get_class_weight_map(self, gt_seg, H, W):
        """Generate class weight map from GT segmentation.

        Args:
            gt_seg: [B, H_gt, W_gt] - ground truth segmentation
            H, W: target feature map size
        Returns:
            weight_map: [B, 1, H, W] - class weight map
        """
        B = gt_seg.shape[0]

        # Resize GT to match feature size if needed
        if gt_seg.shape[1:] != (H, W):
            gt_seg = F.interpolate(
                gt_seg.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()

        # Create class weight map: [B, H, W]
        weight_map = torch.zeros(B, H, W, device=gt_seg.device, dtype=torch.float32)
        for class_idx in range(len(self.class_weights)):
            mask = (gt_seg == class_idx)
            weight_map[mask] = self.class_weights[class_idx]

        return weight_map.unsqueeze(1)  # [B, 1, H, W]

    def forward(self, x, gt_semantic_seg=None):
        """
        Args:
            x: [B, C, H, W] - feature maps
            gt_semantic_seg: [B, H_gt, W_gt] - ground truth segmentation (during training)
        Returns:
            enhanced: [B, C, H, W] - features with class-weighted enhancement
        """
        B, C, H, W = x.shape

        # Edge detection
        edge_map = torch.sigmoid(self.edge_module(x))  # [B, 1, H, W]

        # Detail refinement
        detail_feats = self.detail_module(x)  # [B, C, H, W]

        if self.training and gt_semantic_seg is not None:
            # Training: use GT masks to compute class-weighted enhancement mask
            weight_map = self.get_class_weight_map(gt_semantic_seg, H, W)
        else:
            # Inference: use edge-based enhancement only
            weight_map = torch.ones_like(edge_map)

        # Combine edge detection with class weights
        enhancement_mask = edge_map * weight_map  # [B, 1, H, W]

        # Apply weighted enhancement
        alpha = torch.sigmoid(self.fusion_weight)
        enhanced = x + alpha * enhancement_mask * detail_feats

        return enhanced


@MODELS.register_module()
class UPerHead_HRDE_GTWeighted(UPerHead):
    """UPerHead with Ground Truth Weighted HRDE.

    Addresses UAVid's class imbalance by focusing HRDE on weak classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # GT-weighted HRDE module
        self.hrde = GTWeightedHRDE(
            in_channels=self.channels,
            num_classes=kwargs.get('num_classes', 8)
        )

    def forward(self, inputs, data_samples=None):
        """Forward function with optional GT for training."""

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

        # Extract GT if available (during training)
        gt_seg = None
        if self.training and data_samples is not None:
            gt_seg = self._extract_gt_seg(data_samples)

        # Apply GT-weighted HRDE
        feats = self.hrde(feats, gt_semantic_seg=gt_seg)

        # Final classification
        output = self.cls_seg(feats)

        return output

    def _extract_gt_seg(self, data_samples):
        """Extract GT segmentation from data_samples.

        Args:
            data_samples: list of PixelData containing gt_sem_seg
        Returns:
            gt_seg: [B, H, W] tensor
        """
        gt_segs = []
        for data_sample in data_samples:
            gt_seg = data_sample.gt_sem_seg.data
            gt_segs.append(gt_seg)

        return torch.stack(gt_segs, dim=0).squeeze(1)  # [B, H, W]

    def loss(self, inputs, batch_data_samples, train_cfg):
        """Compute segmentation loss."""
        seg_logits = self.forward(inputs, data_samples=batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
