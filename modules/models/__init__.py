from .base_model import BaseModel
from .classifier import Classifier
from .models import BaseTimmModel
from .encoders import MetaVIT

def get_model(config, num_classes):

    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=num_classes)

    return net