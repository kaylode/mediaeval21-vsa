from theseus.base.models import MODEL_REGISTRY

from .timm_models import *
from .metavit import MetaVIT

MODEL_REGISTRY.register(BaseTimmModel)
MODEL_REGISTRY.register(MetaVIT)