import torch
import torch.utils.data as data

import numpy as np
from .dataset import CSVDataset
from torch.utils.data.sampler import WeightedRandomSampler

def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = np.array([class_weighting[t] for t in labels.squeeze()])
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

class CSVDataLoader(data.DataLoader):
    def __init__(self, root_dir, csv_file, image_size, task, batch_size,  _type='train'):
        self.dataset = CSVDataset(
            root_dir=root_dir, 
            csv_file=csv_file,
            image_size=image_size,
            task=task, _type=_type)

        if task == 'T1' and _type == 'train':
            labels = torch.LongTensor(self.dataset.classes_dist).unsqueeze(1)
            sampler = class_imbalance_sampler(labels)
        else:
            sampler = None

        if _type == 'train':
            drop_last = True
        else:
            drop_last = False
            
        super(CSVDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            collate_fn = self.dataset.collate_fn,
            drop_last=drop_last, 
            sampler=sampler
        )





