import torch.utils.data as data
from .dataset import CSVDataset


class CSVDataLoader(data.DataLoader):
    def __init__(self, root_dir, csv_file, image_size, keep_ratio, task, batch_size,  _type='train'):
        self.dataset = CSVDataset(
            root_dir=root_dir, 
            csv_file=csv_file,
            image_size=image_size,
            keep_ratio=keep_ratio,
            task=task, _type=_type)

        self.collate_fn = self.dataset.collate_fn

        super(CSVDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
        )