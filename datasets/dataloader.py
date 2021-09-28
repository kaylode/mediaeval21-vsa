import os
import torch
import torch.utils.data as data

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataset import CSVDataset


class CSVDataLoader(data.DataLoader):
    def __init__(self, root_dir, csv_file, task, batch_size):
        self.dataset = CSVDataset(
            root_dir=root_dir, 
            csv_file=csv_file,
            task=task)

        self.collate_fn = self.dataset.collate_fn

        super(CSVDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size
        )