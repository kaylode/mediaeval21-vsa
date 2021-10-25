import os
import torch
import torch.utils.data as data
from torchvision.transforms import transforms as tf

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.augmentations.transforms import get_augmentation, Denormalize
from datasets.augmentations.custom import RandomCutmix, RandomMixup


class CSVDataset(data.Dataset):
    """
    - Reads a CSV file. Requires first column to be text data, second column to be labels.
    - Arguments:
   
    """
    
    def __init__(self, root_dir, csv_file, image_size, task='T1', _type='train'):
        super().__init__()

        self.root_dir = root_dir
        self.csv_file = csv_file
        self.task = task

        self.transforms = get_augmentation(image_size, _type=_type)
        
        self.fns, self.classes = self.load_data()
        self.num_classes = len(self.classes)

        # MixUp and CutMix
        mixup_transforms = []
        mixup_transforms.append(RandomMixup(self.num_classes, p=1.0, alpha=0.2))
        mixup_transforms.append(RandomCutmix(self.num_classes, p=1.0, alpha=1.0))
        self.mixupcutmix = tf.RandomChoice(mixup_transforms)

    def load_data(self):
        if self.task == 'T1':
            return self.load_data_t1()
        else:
            return self.load_data_t2()

    def load_data_t2(self):
        df = pd.read_csv(self.csv_file)
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


    def load_data_t1(self):
        self.classes_dist = []
        df = pd.read_csv(self.csv_file)
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
            if classes == 'positive':
                classes = 2
            elif classes == 'negative':
                classes = 1
            else:
                classes = 0
            fns.append((image_name, classes))
            self.classes_dist.append(classes)
            
        return fns, labels
    
    def load_image(self, image_id):
        img_path = os.path.join(self.root_dir, image_id+'.jpg')
        img = Image.open(img_path).convert('RGB')
        aug_img = self.transforms(img)
        return aug_img, img

    def __getitem__(self, index):
        image_id, label = self.fns[index]
        img, ori_img = self.load_image(image_id)
        if not isinstance(label, list):
            label  = torch.LongTensor([label])
        else:
            label  = torch.LongTensor(label)

        return {
            "img" : img,
            'ori_img': ori_img,
            "target" : label}

    def collate_fn(self, batch):
        ori_imgs = [s['ori_img'] for s in batch]
        imgs = torch.stack([s['img'] for s in batch])
        targets = torch.stack([s['target'] for s in batch])

        targets = targets.squeeze(1)
        if self.mixupcutmix is not None:
            imgs, targets = self.mixupcutmix(imgs, targets)

        if self.task != 'T1':
            targets = targets.float()

        return {
            'ori_imgs': ori_imgs,
            'imgs': imgs,
            'targets': targets
        }

    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = np.random.randint(0,len(self.fns))
        item = self.__getitem__(index)
        img = item['img']
        label = item['target']

        # Denormalize and reverse-tensorize
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms:
                if isinstance(x, tf.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            img = denormalize(img = img)

        label = label.numpy()
        self.visualize(img, label, figsize = figsize)

    
    def visualize(self, img, label, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

        if isinstance(img, torch.Tensor):
            img = img.numpy().squeeze().transpose((1,2,0))
            
        # Display the image
        ax.imshow(img)
        if self.task == 'T1':
            plt.title(self.classes[int(label)])
        else:
            title = [self.classes[int(i)] for i, l in enumerate(label) if l==1]
            title = ' '.join(title)
            plt.title(title)
        plt.show()

    def count_dict(self):
        """
        Count for each label
        """

        if self.task == 'T1':
            cnt_dict = {}
            for _, label in self.fns:
                class_name = self.classes[label]
                if class_name in cnt_dict.keys():
                    cnt_dict[class_name] += 1
                else:
                    cnt_dict[class_name] = 1
        else:
            cnt_dict = {}
            for _, label in self.fns:
                for class_name, is_true in zip(self.classes, label):
                    if is_true:
                        if class_name in cnt_dict.keys():
                            cnt_dict[class_name] += 1
                        else:
                            cnt_dict[class_name] = 1

        return cnt_dict

    def plot(self, figsize = (8,8), types = ["freqs"]):
        """
        Dataset Visualization
        """
        cnt_dict = self.count_dict()
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            plt.title("Classes Distribution")
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            plt.xlabel("Classes")
            plt.ylabel("Number of samples")
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')


    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s1 + s2