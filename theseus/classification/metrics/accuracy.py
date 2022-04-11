from typing import Any, Dict, Optional

from theseus.base.metrics.metric_template import Metric
from theseus.classification.utilities.logits import logits2labels

class Accuracy(Metric):
    """Accuracy metric
    """

    def __init__(self, type: str = 'multiclass', **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.reset()

    def update(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        output = output["outputs"] 
        target = batch["targets"] 
        prediction = logits2labels(output, self.type)

        correct = (prediction.view(-1) == target.view(-1)).sum()
        correct = correct.cpu()
        self.total_correct += correct
        self.sample_size += prediction.size(0)

    def value(self):
        return {'acc': (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0