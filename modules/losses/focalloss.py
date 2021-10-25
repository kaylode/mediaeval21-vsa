import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha= 0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)