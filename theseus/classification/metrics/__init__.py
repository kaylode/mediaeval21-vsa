from theseus.base.metrics import METRIC_REGISTRY

from .accuracy import *
from .f1 import *
from .confusion_matrix import *
from .errorcases import *
from .projection import *

METRIC_REGISTRY.register(Accuracy)
METRIC_REGISTRY.register(F1ScoreMetric)
METRIC_REGISTRY.register(ConfusionMatrix)
METRIC_REGISTRY.register(ErrorCases)
METRIC_REGISTRY.register(EmbeddingProjection)

from .multilabel.accuracy import MultilabelAccuracy
from .multilabel.f1 import MultilabelF1ScoreMetric
from .multilabel.confusion_matrix import MultilabelConfusionMatrix

METRIC_REGISTRY.register(MultilabelAccuracy)
METRIC_REGISTRY.register(MultilabelF1ScoreMetric)
METRIC_REGISTRY.register(MultilabelConfusionMatrix)