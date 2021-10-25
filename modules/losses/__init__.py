import torch.nn as nn
from .focalloss import FocalLoss

def get_loss(name, **kwargs):
    if name == 'focal':
        return FocalLoss()
    if name == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == 'bce':
        return nn.BCEWithLogitsLoss()