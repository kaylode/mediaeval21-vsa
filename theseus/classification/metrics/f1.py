from sklearn.metrics import f1_score
from typing import Any, Dict, Optional

from theseus.base.metrics.metric_template import Metric
from theseus.classification.utilities.logits import logits2labels

class F1ScoreMetric(Metric):
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'weighted', type:str = 'multiclass', **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.type =type
        self.threshold = kwargs.get('threshold', 0.5)
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"] 
        outputs = outputs["outputs"] 

        outputs = logits2labels(outputs, type=self.type, threshold=self.threshold)
        targets = targets.squeeze()
    
        self.preds +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()

    def value(self):
        score = f1_score(self.targets, self.preds, average=self.average, zero_division=0)
        return {f"{self.average}-f1": score}

    def reset(self):
        self.targets = []
        self.preds = []