import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from .lib_upernet.nn import SynchronizedBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        SynchronizedBatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class UncertaintyGuidedRefinement(nn.Module):
    """Uncertainty-guided Adaptive Refinement (UAR) module.

    Given coarse prediction and fused features, computes an uncertainty map
    (entropy of softmax), then refines features in high-uncertainty regions
    using deformable convolution with offsets guided by the uncertainty map.
    """

    def __init__(self, feat_channels, num_classes, uncertainty_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold

        # Learnable threshold
        self.threshold_param = nn.Parameter(torch.tensor(0.0))

        # Offset prediction from uncertainty + features
        self.offset_conv = nn.Sequential(
            nn.Conv2d(feat_channels + 1, feat_channels // 4, 3, padding=1, bias=False),
            SynchronizedBatchNorm2d(feat_channels // 4),
            nn.ReLU(inplace=True),
        )
        # 3x3 deformable conv: 9 points * 2 (offset) + 9 points * 1 (mask) = 27
        self.offset_pred = nn.Conv2d(feat_channels // 4, 27, 3, padding=1, bias=True)
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)

        # Deformable conv for refinement
        self.deform_conv = ModulatedDeformConv2d(
            feat_channels, feat_channels, 3, padding=1, bias=False)
        self.deform_bn = SynchronizedBatchNorm2d(feat_channels)
        self.deform_relu = nn.ReLU(inplace=True)

        # Channel attention on refined features
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_channels, feat_channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Final prediction from refined features
        self.refine_pred = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1, bias=False),
            SynchronizedBatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, num_classes, 1, bias=True),
        )

    def compute_uncertainty(self, logits):
        """Compute pixel-wise uncertainty as normalized entropy."""
        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1, keepdim=True)
        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(float(self.num_classes), device=logits.device))
        uncertainty = entropy / (max_entropy + 1e-8)
        return uncertainty

    def forward(self, features, coarse_logits):
        """
        Args:
            features: fused feature map [B, C, H, W]
            coarse_logits: initial prediction [B, num_classes, H, W]
        Returns:
            refined_logits: [B, num_classes, H, W]
            uncertainty_map: [B, 1, H, W]
        """
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(coarse_logits.detach())

        # Adaptive threshold via sigmoid
        threshold = torch.sigmoid(self.threshold_param) * 0.8 + 0.1  # range [0.1, 0.9]

        # Soft mask: higher weight for more uncertain pixels
        soft_mask = torch.clamp((uncertainty - threshold) / (1.0 - threshold + 1e-8), 0, 1)

        # Predict deformable offsets guided by uncertainty
        offset_input = torch.cat([features, uncertainty], dim=1)
        offset_feat = self.offset_conv(offset_input)
        offset_out = self.offset_pred(offset_feat)

        # Split into offset and mask for modulated deformable conv
        offset = offset_out[:, :18, :, :]
        mask = torch.sigmoid(offset_out[:, 18:, :, :])

        # Deformable convolution refinement
        refined = modulated_deform_conv2d(
            features, offset, mask, self.deform_conv.weight, self.deform_conv.bias,
            1, 1, 1, 1, 1)
        refined = self.deform_relu(self.deform_bn(refined))

        # Channel attention
        ca = self.channel_attn(refined)
        refined = refined * ca

        # Selective fusion: only refine uncertain regions
        fused_features = features + soft_mask * (refined - features)

        # Predict refined logits
        refined_logits = self.refine_pred(fused_features)

        return refined_logits, uncertainty


@MODELS.register_module()
class UPerHead_UAR(BaseDecodeHead):
    """UPerNet decode head with Uncertainty-guided Adaptive Refinement.

    Two-stage prediction:
    1. Standard UPerNet FPN fusion -> coarse prediction
    2. UAR module refines uncertain regions -> final prediction
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), pretrained=None, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        # PPM Module
        self.ppm_pooling = nn.ModuleList()
        self.ppm_conv = nn.ModuleList()
        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(self.in_channels[-1], self.channels, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ))
        self.ppm_last_conv = conv3x3_bn_relu(
            self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 1)

        # FPN Module
        self.fpn_in = nn.ModuleList()
        for fpn_inplane in self.in_channels[:-1]:
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, self.channels, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ))

        self.fpn_out = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.fpn_out.append(conv3x3_bn_relu(self.channels, self.channels, 1))

        self.conv_fusion = conv3x3_bn_relu(
            len(self.in_channels) * self.channels, self.channels, 1)

        # Coarse prediction head
        self.coarse_head = nn.Sequential(
            conv3x3_bn_relu(self.channels, self.channels, 1),
            nn.Conv2d(self.channels, self.out_channels, kernel_size=1, bias=True)
        )

        # UAR module
        self.uar = UncertaintyGuidedRefinement(
            feat_channels=self.channels,
            num_classes=self.out_channels,
        )

        self.pretrained = pretrained
        self._init_weights()

    def _init_weights(self):
        if self.pretrained is not None:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            from collections import OrderedDict
            new_ckpt = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module'):
                    new_k = k.replace('module.', '')
                else:
                    new_k = k
                if new_k.startswith('backbone'):
                    continue
                if new_k.startswith('decode_head'):
                    new_k = new_k.replace('decode_head.', '')
                if new_k.startswith('auxiliary_head'):
                    continue
                new_ckpt[new_k] = v
            state_dict = new_ckpt

            # Map old keys to new structure
            mapped_ckpt = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('object_head.'):
                    new_k = k.replace('object_head.', 'coarse_head.')
                    mapped_ckpt[new_k] = v
                else:
                    mapped_ckpt[new_k] = v

            load_state_dict(self, mapped_ckpt, strict=False, logger=None)

    def _fuse_features(self, x):
        """FPN feature fusion, returns fused feature map."""
        seg_size = x[-1].shape[-2:]

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

        fpn_feature_list = [f]
        for i in reversed(range(len(x) - 1)):
            conv_x = x[i]
            conv_x = self.fpn_in[i](conv_x)
            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i], output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        fused = self.conv_fusion(fusion_out)

        return fused, seg_size

    def forward(self, x):
        """Forward pass returning final refined logits."""
        fused, seg_size = self._fuse_features(x)

        # Stage 1: coarse prediction
        coarse_logits = self.coarse_head(fused)

        # Stage 2: uncertainty-guided refinement
        refined_logits, uncertainty = self.uar(fused, coarse_logits)

        # Use refined logits as final output
        return F.interpolate(refined_logits, size=seg_size, mode='bilinear', align_corners=False)

    def forward_train(self, x):
        """Forward pass for training, returns both coarse and refined logits."""
        fused, seg_size = self._fuse_features(x)

        coarse_logits = self.coarse_head(fused)
        refined_logits, uncertainty = self.uar(fused, coarse_logits)

        # Store for loss computation
        self._coarse_logits = F.interpolate(
            coarse_logits, size=seg_size, mode='bilinear', align_corners=False)
        self._uncertainty = F.interpolate(
            uncertainty, size=seg_size, mode='bilinear', align_corners=False)

        return F.interpolate(refined_logits, size=seg_size, mode='bilinear', align_corners=False)
