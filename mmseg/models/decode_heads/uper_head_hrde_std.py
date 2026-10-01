"""
UPerHead_HRDE_std: HRDE on top of standard UPerHead.

与原 UPerHead_HRDE 的区别：
- 父类换成标准 UPerHead（不是 UPerHead_lwc）
- 模块层 key 与标准 UPerHead checkpoint 完全一致 -> 可以从 SP 训练好的
  checkpoint 完整加载 head 权重
- 仅 hr_enhancement 模块从零初始化
HRDE 模块本身（EdgeAware + DetailRefinement + 自适应融合）与原版一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .uper_head import UPerHead


class EdgeAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            ConvModule(in_channels, in_channels // 2, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels // 2, in_channels // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.edge_conv(x))


class DetailRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.detail_conv1 = ConvModule(
            in_channels, in_channels // 2, kernel_size=3, padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        self.detail_conv2 = ConvModule(
            in_channels, in_channels // 2, kernel_size=5, padding=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        self.fusion = ConvModule(
            in_channels, in_channels, kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        d1 = self.detail_conv1(x)
        d2 = self.detail_conv2(x)
        return self.fusion(torch.cat([d1, d2], dim=1))


class HighResolutionEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_module = EdgeAwareModule(in_channels)
        self.detail_module = DetailRefinementModule(in_channels)
        self.fusion_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        edge_attn = self.edge_module(x)
        detail = self.detail_module(x)
        w = torch.sigmoid(self.fusion_weight)
        return x + w * edge_attn * detail


@MODELS.register_module()
class UPerHead_HRDE_std(UPerHead):
    """Standard UPerHead + HRDE enhancement before final classifier.

    Forward: backbone feats -> _forward_feature (PSP+FPN, 父类继承)
             -> hr_enhancement -> cls_seg
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hr_enhancement = HighResolutionEnhancement(in_channels=self.channels)

    def forward(self, inputs):
        feats = self._forward_feature(inputs)        # [B, C, H, W]
        feats = self.hr_enhancement(feats)            # 仅这里是新增模块
        return self.cls_seg(feats)
