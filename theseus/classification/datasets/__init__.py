from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .csv_dataset import *

DATASET_REGISTRY.register(CSVDataset)
DATASET_REGISTRY.register(VSA_T1)
DATASET_REGISTRY.register(VSA_T2)

from .mixupcutmix_collator import MixupCutmixCollator

DATALOADER_REGISTRY.register(MixupCutmixCollator)