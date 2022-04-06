from typing import List, Optional, Dict
import torch
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .dataset import ClassificationDataset

from theseus.classification.utilities.batch import make_feature_batch
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')

class CSVDataset(ClassificationDataset):
    r"""CSVDataset multi-labels classification dataset

    Reads in .csv file with structure below:
        filename | label
        -------- | -----

    image_dir: `str`
        path to directory contains images
    csv_path: `str`
        path to csv file
    txt_classnames: `str`
        path to txt file contains classnames
    transform: Optional[List]
        transformatin functions
    test: bool
        whether the dataset is used for training or test
        
    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        face_dir: str,
        det_dir: str,
        transform: Optional[List] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.face_dir = face_dir
        self.det_dir = det_dir
        self.csv_path = csv_path
        self.transform = transform
        self.load_data()
    
    def load_numpy(self, image_id):
        face_path = osp.join(self.face_dir, image_id+'.npz')
        det_path = osp.join(self.det_dir, image_id+'.npz')
        box_path = osp.join(self.det_dir, image_id+'_loc.npz')

        if osp.exists(face_path):
            face_npy = np.load(face_path)['feat']
        else:
            face_npy = np.zeros((1,512))

        det_npy = np.load(det_path)['arr_0']
        box_npy = np.load(box_path)['arr_0']
        return face_npy, det_npy, box_npy

    def load_data(self):
        """
        Read data from csv and load into memory
        """

        self.fns, self.classnames = self._load_data()

        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

    def collate_fn(self, batch: List):
        """
        Collator for wrapping a batch
        """
        imgs = torch.stack([s['input'] for s in batch])
        targets = torch.stack([torch.LongTensor(s['target']['labels']) for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_sizes = [s['ori_size'] for s in batch]

        npy_faces = [s['facial_feat'] for s in batch]
        npy_dets = torch.stack([s['det_feat'] for s in batch])
        npy_boxes = torch.stack([s['loc_feat'] for s in batch])

        npy_faces = make_feature_batch(npy_faces, pad_token=0)


        if self.task != 'T1':
            targets = targets.float()

        return {
            'inputs': imgs,
            'targets': targets,
            'img_names': img_names,
            'ori_sizes': ori_sizes,

            'facial_feats': npy_faces.float(),
            'det_feats': npy_dets.float(),
            'loc_feats': npy_boxes.float(),
        }

class VSA_T1(CSVDataset):
    r"""CSVDataset multi-labels classification dataset

    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        face_dir: str,
        det_dir: str,
        transform: Optional[List] = None,
        **kwargs
    ):
        self.task = 'T1'
        super().__init__(image_dir, csv_path, face_dir, det_dir, transform)

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        fns = []

        colnames = list(df.columns)
        colnames = [i for i in colnames if i=='T1']
        labels = ['neutral', 'negative', 'positive']
        
        colnames.append('id')
        df = df[colnames]

        for _, row in df.iterrows():
            lst = row.tolist()
            image_name = lst[-1]
            classes = lst[0]
            fns.append((image_name, classes))
            
        return fns, labels

    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []
        for _, label_name in self.fns:
            self.classes_dist.append(self.classes_idx[label_name])
        return self.classes_dist

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        image_id, label_name = self.fns[idx]
        face_npy, det_npy, box_npy = self.load_numpy(image_id)

        image_name = image_id +'.jpg'
        image_path = osp.join(self.image_dir, image_name)

        det_tensor = torch.from_numpy(det_npy)
        box_tensor = torch.from_numpy(box_npy)

        im = Image.open(image_path).convert('RGB')

        width, height = im.width, im.height
        class_idx = self.classes_idx[label_name]

        if self.transform:
            im = self.transform(im)

        target = {}
        target['labels'] = [class_idx]
        target['label_name'] = label_name

        return {
            "input": im, 
            "facial_feat" : face_npy,
            'det_feat': det_tensor,
            'loc_feat': box_tensor,
            'target': target,
            'img_name': image_name,
            'ori_size': [width, height]
        }


class VSA_T2(CSVDataset):
    r"""CSVDataset multi-labels classification dataset

    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        face_dir: str,
        det_dir: str,
        transform: Optional[List] = None,
        task: str = 'T2',
        **kwargs
    ):
        self.task = task
        super().__init__(image_dir, csv_path, face_dir, det_dir, transform)

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        fns = []

        colnames = list(df.columns)
        colnames = [i for i in colnames if i.split('.')[0]==self.task]
        labels = [i.split(':')[1].rstrip().lstrip().lower() for i in colnames]

        colnames.append('id')
        df = df[colnames]

        if self.task == 'T2':
            df2 = df[["T2.1: Joy","T2.2: Sadness","T2.3: Fear","T2.4: Disgust","T2.5: Anger","T2.6: Surprise","T2.7: Neutral"]]
            df2['sum'] = df2.sum(axis=1)
            df = df[df2['sum']!=0]

        if self.task == 'T3':
            df3 = df[["T3.1: Anger","T3.2: Anxiety","T3.3: Craving","T3.4: Emphatic pain","T3.5: Fear","T3.6: Horror","T3.7: Joy","T3.8: Relief","T3.9: Sadness","T3.10:surprise"]]
            df3['sum'] = df3.sum(axis=1)
            df = df[df3['sum']!=0]

        for _, row in df.iterrows():
            lst = row.tolist()
            image_name = lst[-1]
            classes = lst[:-1]
            fns.append((image_name, classes))

        return fns, labels

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        image_id, label_ids = self.fns[idx]
        face_npy, det_npy, box_npy = self.load_numpy(image_id)

        image_name = image_id +'.jpg'
        image_path = osp.join(self.image_dir, image_name)

        det_tensor = torch.from_numpy(det_npy)
        box_tensor = torch.from_numpy(box_npy)

        im = Image.open(image_path).convert('RGB')

        width, height = im.width, im.height

        if self.transform:
            im = self.transform(im)

        target = {}
        target['labels'] = label_ids
        target['label_name'] = [name for i, name in zip(label_ids, self.classnames) if i==1]

        return {
            "input": im, 
            "facial_feat" : face_npy,
            'det_feat': det_tensor,
            'loc_feat': box_tensor,
            'target': target,
            'img_name': image_name,
            'ori_size': [width, height]
        }