from typing import Any, Dict, Optional
import torch
import numpy as np
from sklearn.metrics import f1_score

from theseus.base.metrics.metric_template import Metric


class MultilabelF1ScoreMetric(Metric):
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, threshold: float, average = 'weighted', **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.threshold = threshold
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"] 
        outputs = outputs["outputs"] 

        outputs = torch.sigmoid(torch.from_numpy(outputs))
        outputs = outputs > self.threshold

        outputs = outputs.detach().cpu()
        targets = targets.detach().cpu().view(-1)
    
        self.preds +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()

    def value(self):
        score = f1_score(self.targets, self.preds, average=self.average)
        return {f"{self.average}-f1": score}

    def reset(self):
        self.targets = []
        self.preds = []
