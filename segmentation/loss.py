
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=0.97, gamma=2, num_classes=2, activation=True, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class DiceLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=0.99, gamma=2, weight=None, activation=True, reduction="mean", smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.activation = activation
        self.smooth = smooth
        self.weight = weight

    def forward(self, inputs, targets):
        if self.activation:
            probs = torch.sigmoid(inputs)      # Flatten the tensors
        else:
            probs = inputs
        probs_flat = probs.reshape(probs.size(0), probs.size(1), -1)
        targets_flat = targets.reshape(targets.size(0), targets.size(1), -1)

        # Compute intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        # Compute Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Weight by channels
        weighted_dice_score = (dice_score * self.weight.to(dice_score.device)).sum(dim=1)

        # Compute Dice loss
        dice_loss = 1.0 - weighted_dice_score.mean()

        return dice_loss

class Deep_Supervised_Loss(nn.Module):
    def __init__(self, mode='FocalDice', activation=True):
        super(Deep_Supervised_Loss, self).__init__()
        self.fl = FocalLoss(activation=activation)
        self.dl = DiceLoss(activation=activation, weight=torch.tensor([0.2, 0.2, 0.2, 0.4], dtype=torch.float32))
        self.mode = mode
    def forward(self, input, target):
        if self.mode == 'FocalDice':
            return self.dl(input, target) + self.fl(input, target)
        elif self.mode == 'Focal':
            return self.fl(input, target)
        else:
            raise NotImplementedError
