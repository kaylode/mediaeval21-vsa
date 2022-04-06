from typing import Any, Dict, Optional

import torch
from theseus.base.metrics.metric_template import Metric

class MultilabelBalancedAccuracy(Metric):
    """
    Accuracy metric for multilabel classification
    """
    def __init__(self, thresh=0.5, **kwargs):
        super().__init__()
        self.thresh = thresh
        self.reset()
        
    def update(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        outputs = output["outputs"] 
        targets = batch["targets"] 

        outputs = torch.sigmoid(outputs)
        prediction = output > self.thresh
        prediction = prediction.long()

        correct = (prediction.view(-1) == targets.view(-1)).sum()
        correct = correct.cpu()
        self.total_correct += correct
        self.sample_size += prediction.size(0)

    def value(self):
        return {'acc': (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0