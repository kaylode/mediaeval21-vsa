from .focalloss import WeightedFocalLoss
from .smoothceloss import smoothCELoss
import torch.nn as nn

def get_loss(name):
    if name == 'focal':
        return WeightedFocalLoss()
    if name == 'smoothce':
        return smoothCELoss()
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name == 'bce':
        return nn.BCEWithLogitsLoss()