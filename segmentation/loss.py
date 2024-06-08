import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=0.99, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        ce_loss_1 = F.binary_cross_entropy(inputs[:, 0], targets[:, 0], reduction="none")
        ce_loss_2 = F.binary_cross_entropy(inputs[:, 1], targets[:, 1], reduction="none")

        pt_1 = torch.exp(-ce_loss_1)
        pt_2 = torch.exp(-ce_loss_2)

        loss_1 = self.alpha * (1 - pt_1) ** self.gamma * ce_loss_1
        loss_2 = self.alpha * (1 - pt_2) ** self.gamma * ce_loss_2

        if self.reduction == "mean":
            loss_1 = loss_1.mean()
            loss_2 = loss_2.mean()
            return (loss_1 + loss_2) / 2

        elif self.reduction == "sum":
            loss_1 = loss_1.sum()
            loss_2 = loss_2.sum()
            return loss_1 + loss_2

class Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()
    def forward(self, input, target):
        return self.loss(input, target)