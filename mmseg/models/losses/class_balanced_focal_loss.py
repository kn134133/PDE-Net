import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def class_balanced_focal_loss(pred,
                               target,
                               weight=None,
                               gamma=2.0,
                               alpha=0.25,
                               class_weight=None,
                               reduction='mean',
                               avg_factor=None,
                               ignore_index=255):
    """Class-Balanced Focal Loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, H, W).
        target (torch.Tensor): The ground truth with shape (N, H, W).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float): The gamma for calculating the modulating factor.
        alpha (float): A balanced form for Focal Loss.
        class_weight (torch.Tensor, optional): Weight for each class.
        reduction (str): The method to reduce the loss.
        avg_factor (int, optional): Average factor for loss.
        ignore_index (int): The label index to be ignored.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Apply softmax to get probabilities
    pred_softmax = F.softmax(pred, dim=1)

    # Get number of classes
    num_classes = pred.shape[1]

    # One-hot encode target
    target_one_hot = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    # Get the probability of the true class
    pt = (pred_softmax * target_one_hot).sum(dim=1)

    # Focal weight: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma

    # Cross entropy loss
    ce_loss = F.cross_entropy(
        pred, target, reduction='none', ignore_index=ignore_index
    )

    # Apply focal weight
    loss = focal_weight * ce_loss

    # Apply alpha balancing
    if alpha is not None:
        alpha_t = alpha * target_one_hot[:, 1] + (1 - alpha) * target_one_hot[:, 0]
        loss = alpha_t * loss

    # Apply class weight
    if class_weight is not None:
        class_weight_map = torch.zeros_like(target, dtype=torch.float)
        for i, w in enumerate(class_weight):
            class_weight_map[target == i] = w
        loss = loss * class_weight_map

    # Create valid mask (ignore ignore_index)
    valid_mask = (target != ignore_index).float()
    loss = loss * valid_mask

    # Apply sample weight
    if weight is not None:
        loss = loss * weight

    # Reduce loss
    if reduction == 'mean':
        if avg_factor is None:
            avg_factor = valid_mask.sum() + 1e-6
        loss = loss.sum() / avg_factor
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


@MODELS.register_module()
class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for addressing class imbalance.

    This loss combines:
    1. Focal Loss: focuses on hard examples
    2. Class balancing: weights classes by inverse frequency

    Args:
        gamma (float): Focusing parameter for modulating loss. Default: 2.0.
        alpha (float): Balancing parameter. Default: 0.25.
        use_class_weight (bool): Whether to use class-balanced weighting.
        class_weight (list, optional): Manual class weights.
        reduction (str): The method to reduce the loss. Default: 'mean'.
        loss_weight (float): Weight of the loss. Default: 1.0.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 use_class_weight=True,
                 class_weight=None,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_class_weight = use_class_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        # Statistics for class balancing
        self.register_buffer('class_counts', torch.zeros(2))
        self.register_buffer('total_samples', torch.tensor(0))

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return 'loss_cb_focal'

    def update_class_stats(self, target):
        """Update class statistics for dynamic class balancing."""
        valid_mask = (target != self.ignore_index)
        for i in range(2):
            count = (target[valid_mask] == i).sum()
            self.class_counts[i] += count
        self.total_samples += valid_mask.sum()

    def get_class_weight(self):
        """Compute class weights based on inverse frequency."""
        if self.class_weight is not None:
            return torch.tensor(self.class_weight, device=self.class_counts.device)

        if not self.use_class_weight or self.total_samples == 0:
            return None

        # Inverse frequency weighting
        class_freq = self.class_counts / (self.total_samples + 1e-6)
        class_weight = 1.0 / (class_freq + 1e-6)
        # Normalize so that weights sum to num_classes
        class_weight = class_weight / class_weight.sum() * 2

        return class_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C, H, W).
            target (torch.Tensor): The ground truth with shape (N, H, W).
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor for loss.
            reduction_override (str, optional): Override reduction method.
            ignore_index (int, optional): The label index to be ignored.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # Use provided ignore_index or default
        if ignore_index is None:
            ignore_index = self.ignore_index

        # Update class statistics during training
        if self.training and self.use_class_weight:
            self.update_class_stats(target)

        # Get class weight
        class_weight = self.get_class_weight()

        # Calculate loss
        loss = self.loss_weight * class_balanced_focal_loss(
            pred,
            target,
            weight=weight,
            gamma=self.gamma,
            alpha=self.alpha,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index
        )

        return loss
