import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor
from monai.losses import DiceLoss

from utils import one_hot_encode

class BinaryFocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=0.5, gamma=1, num_classes=2, activation=True, reduction="sum"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        # ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        #
        # pt = torch.exp(-ce_loss)
        # loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [getattr(self, k) for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        return f'{type(self).__name__}({", ".join(arg_strs)})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        if unignored_mask.sum() == 0:
            return torch.tensor(0., dtype=x.dtype, device=x.device)

        x = x[unignored_mask]
        y = y[unignored_mask]

        log_p = F.log_softmax(x, dim=-1)
        pt = log_p[torch.arange(len(y)), y].exp()
        focal_term = (1 - pt)**self.gamma

        # Dynamically create NLLLoss with alpha moved to correct device
        nll_loss = nn.NLLLoss(
            weight=self.alpha.to(x.device) if self.alpha is not None else None,
            reduction='none',
            ignore_index=self.ignore_index
        )
        ce_loss = nll_loss(log_p, y)

        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class Binary_Deep_Supervised_Loss(nn.Module):
    def __init__(self, mode='FocalDice', activation=True):
        super(Binary_Deep_Supervised_Loss, self).__init__()
        self.fl = BinaryFocalLoss(reduction='sum', activation=activation)
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

class Deep_Supervised_Loss(nn.Module):
    def __init__(self, mode='FocalDice', activation=True, alpha=None):
        super(Deep_Supervised_Loss, self).__init__()
        self.fl = FocalLoss(gamma=2, alpha=alpha, reduction='sum')
        self.dl = DiceLoss(
            include_background=False,  # often better for brain MRI
            to_onehot_y=True,          # turns integer targets into one-hot internally
            softmax=True,              # applies softmax to logits internally (multi-class)
            squared_pred=True,         # smoother gradients, helps stability
            reduction="mean",          # "mean" | "sum" | "none"
            smooth_nr=0.0,             # numerator smoothing
            smooth_dr=1e-5,            # denominator smoothing
            batch=True,
        )
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