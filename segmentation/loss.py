import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute the focal loss
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()
    def forward(self, input, target):
        return self.loss(input, target)