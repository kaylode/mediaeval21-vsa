import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

class F1ScoreMetric():
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'macro'):
        self.average = average
        self.reset()

    def update(self, outputs, targets):
        pred = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        self.targets.append(targets)
        self.preds.append(pred)

    def compute(self): 
        self.targets = np.concatenate(self.targets)
        self.preds = np.concatenate(self.preds)
        return f1_score(self.targets, self.preds, average=self.average)

    def value(self):
        score = self.compute()
        return {"f1-score" : score}

    def reset(self):
        self.targets = []
        self.preds = []
    
    def __str__(self):
        return f'F1-Score: {self.value()}'

if __name__ == '__main__':
    f1 = F1ScoreMetric()
    y_true = torch.tensor([[1,0,1],[1,1,0],[0,0,0]])
    y_pred = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    f1.update(y_pred ,y_true)
    f1.update(y_pred ,y_true)
    print(f1.value())

