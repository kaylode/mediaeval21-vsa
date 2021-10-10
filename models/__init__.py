from .base_model import BaseModel
from .classifier import Classifier
from .models import BaseTimmModel
from .vit import TransformerBasedModel


def get_model(config, num_classes):

    if config.model_name.startswith('vit'):
        net = TransformerBasedModel(name='vit', num_classes=num_classes)

    elif config.model_name.startswith('twinssvt'):
        net = TransformerBasedModel(name='twinssvt', num_classes=num_classes)

    else:
        net = BaseTimmModel(
            name=config.model_name, 
            num_classes=num_classes)

    return net