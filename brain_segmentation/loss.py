import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor

from utils import one_hot_encode


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
class DiceLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=0.99, gamma=2, num_classes=4, activation=True, reduction="mean", smooth=1.0):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes
        self.activation = activation
        self.smooth = smooth

    def forward(self, inputs, targets):
        logits = torch.softmax(inputs, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Ensure targets are of type long
        targets = targets.long()

        # One-hot encode the predictions and targets
        preds_one_hot = one_hot_encode(preds, self.num_classes)
        targets_one_hot = one_hot_encode(targets, self.num_classes)

        # Compute the intersection and union for each class
        intersection = (preds_one_hot * targets_one_hot).sum(dim=(0, 2, 3))
        union = preds_one_hot.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        # Compute the Dice coefficient for each class
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Compute the Dice loss for each class
        dice_loss = 1.0 - dice_coeff
        return torch.mean(dice_loss)

class Deep_Supervised_Loss(nn.Module):
    def __init__(self, mode='FocalDice', activation=True):
        super(Deep_Supervised_Loss, self).__init__()
        self.fl = FocalLoss(gamma=2, alpha=torch.Tensor([0.01, 1, 1, 1]), reduction='sum')
        self.dl = DiceLoss(activation=activation)
        self.mode = mode
    def forward(self, input, target):
        if self.mode == 'FocalDice':
            return self.dl(input, target) + self.fl(input, target)
        elif self.mode == 'Focal':
            return self.fl(input, target)
        elif self.mode == 'Dice':
            return self.dl(input, target)
        else:
            raise NotImplementedError