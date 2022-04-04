from theseus.base.metrics import METRIC_REGISTRY

from .accuracy import *
from .bl_accuracy import *
from .f1 import *
from .confusion_matrix import *
from .errorcases import *
from .projection import *

METRIC_REGISTRY.register(Accuracy)
METRIC_REGISTRY.register(BalancedAccuracyMetric)
METRIC_REGISTRY.register(F1ScoreMetric)
METRIC_REGISTRY.register(ConfusionMatrix)
METRIC_REGISTRY.register(ErrorCases)
METRIC_REGISTRY.register(EmbeddingProjection)

from .multilabel.accuracy import MultilabelAccuracy
from .multilabel.f1 import MultilabelF1ScoreMetric
from .multilabel.bl_accuracy import MultilabelBalancedAccuracy

METRIC_REGISTRY.register(MultilabelAccuracy)
METRIC_REGISTRY.register(MultilabelBalancedAccuracy)
METRIC_REGISTRY.register(MultilabelF1ScoreMetric)