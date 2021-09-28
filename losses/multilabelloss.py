import torch.nn as nn
import torch


class MultiLabelsLoss(nn.Module):
    """
    References: https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
    """
    def __init__(self):
        super(MultiLabelsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, outputs, targets):
        # Outputs size: batch_size * num_classes
        # Targets size: batch_size

        loss = self.criterion(outputs, targets)
  
        return loss, {'T': loss.item()}
        