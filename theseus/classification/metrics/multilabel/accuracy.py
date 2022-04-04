from typing import Any, Dict, Optional

import torch
from theseus.base.metrics.metric_template import Metric

class MultilabelAccuracy(Metric):
    """
    Accuracy metric for multilabel classification
    """
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.reset()
        
    def update(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        outputs = output["outputs"] 
        targets = batch["targets"] 

        outputs = torch.sigmoid(outputs)
        prediction = outputs > self.threshold
        prediction = prediction.long().detach().cpu()

        correct = (prediction == targets)

        results = 0
        for i in correct:
            results += i.all()

        self.total_correct += results
        self.sample_size += prediction.size(0)

    def value(self):
        return {'acc': (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0