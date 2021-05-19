""" Waterbird dataset with place backgrounds
Assumes dataset comes from this tarball:
https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
[from this link: https://github.com/kohpangwei/group_DRO]
"""
import os
import pandas as pd
import numpy as np
from PIL import Image


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
import pytorch_lightning as pl

DATASET_DIR = os.path.expanduser('~/datasets') # CHANGE THIS PER YOUR NEEDS

# =================================================================
# =           Torch Dataset Method                                =
# =================================================================

class BirdSet(Dataset):
    def __init__(self, root_dir=DATASET_DIR, transform=None, useful_getitem=True):
        self.bird_dir = os.path.join(root_dir, 'waterbird_complete95_forest2water2')
        self.metadata = pd.read_csv(os.path.join(self.bird_dir, 'metadata.csv'))

        self.transform = transform
        if transform is None:
            # Default transform
            self.transform = transforms.Compose([
                       transforms.RandomResizedCrop(
                           (224, 224),
                           scale=(0.7, 1.0),
                           ratio=(0.75, 1.3333333333333333),
                           interpolation=2),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()])

        # Collect y values
        self.y_array = self.metadata['y'].values
        # Y=0 => Landbird
        # Y=1 => Waterbird
        self.n_classes = 2

        # Collect confounding attributes
        self.confounder_array = self.metadata['place'].values
        # A=0 => Land background
        # A=1 => Water background
        self.n_confounders = 1

        # Collect groups: G(a,y) = 2y + a
        self.group_array = (self.y_array *2 + self.confounder_array).astype('int')
        # G=0 => Land bird on land background (majority)
        # G=1 => Land bird on water background (minority)
        # G=2 => Waterbird on land background (minority)
        # G=3 => Waterbird on water background (majority)

        self.filename_array = self.metadata['img_filename'].values
        self.split_array = self.metadata['split'].values
        self.split_dict = {'train':0, 'val': 1, 'test': 2}

        self.useful_getitem = useful_getitem

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        a = self.confounder_array[idx]

        img_name = os.path.join(self.bird_dir, self.metadata.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        sample = {'image': image, 'target': y,
                  'group': g, 'spurious': a}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.useful_getitem:
            return sample['image'], sample['target']
        return sample


    def make_subset(self, filter_dict):
        """ Returns a subset of this dataset according to the filter_dict
        e.g. {split: [train, val, ],
              group: [...]}
        """
        valid_filter_keys = ['group', 'split']
        assert all([k in valid_filter_keys] for k in filter_dict)

        # Group filter first
        group_idxs = None
        if 'group' in filter_dict:
            groups = filter_dict['group']
            if isinstance(groups, int):
                groups = [groups]
            group_idxs = set([i for i, _ in enumerate(self.group_array)
                              if _ in groups])

        # Split
        split_idxs = None
        if 'split' in filter_dict:
            split_val = self.split_dict[filter_dict['split']]
            split_idxs = set([i for i, _ in enumerate(self.split_array)
                              if _ == split_val])

        # And then take conjunctions:
        if group_idxs is None:
            idxs = split_idxs
        elif split_idxs is None:
            idxs = group_idxs
        else:
            idxs = split_idxs.intersection(group_idxs)

        return Subset(self, np.array(list(idxs)))



# =================================================================
# =           Pytorch Lightning DataModule                        =
# =================================================================

class BirdModule(pl.LightningDataModule):
    def __init__(self, root_dir=DATASET_DIR, transform=None, groups=None,
                 batch_size=128):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.groups = groups
        self.batch_size = batch_size

    def setup(self, stage=None):
        filter_dict = {}
        if self.groups is not None:
            filter_dict['group'] = self.groups
        full_data = BirdSet(root_dir=self.root_dir, transform=self.transform)


        filter_dict['split'] = 'train'
        self.train_data = full_data.make_subset(filter_dict)
        filter_dict['split'] = 'val'
        self.val_data = full_data.make_subset(filter_dict)
        filter_dict['split'] = 'test'
        self.test_data = full_data.make_subset(filter_dict)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

