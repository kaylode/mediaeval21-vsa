import torch.nn as nn
from .focalloss import FocalLoss

def get_loss(name, **kwargs):
    if name == 'focal':
        return FocalLoss()
    if name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    if name == 'bce':
        return nn.BCEWithLogitsLoss()