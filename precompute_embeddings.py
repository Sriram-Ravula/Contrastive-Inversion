#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import torch
from torch import Tensor
import typing
import torch.nn.functional as F
import model
import clip
import copy
import pickle
from tqdm import tqdm

import torch
import torchvision

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.metrics import Accuracy
from torch.utils.data  import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import CIFAR10

def precompute_embeddings():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    #args.plugins = DDPPlugin(find_unused_parameters=False)

    seed_everything(args.seed)

    train_data = ImageNet100(
        root=self.hparams.dataset_dir,
        split="train",
        transform=None
    )
    train_dl = DataLoader(train_data, transform=ImageNetBaseTransform(self.hparams), batch_size=256, shuffle=False)

    baseclip = clip.load(self.hparams.baseclip_type, self.hparams.device)[0]
    baseclip.eval()
    embeds_list = []
    with torch.no_grad():
        for images, _ in train_dl:
            embeds = baseclip.encode_image(images)
            embeds_list.append(embeds)

    all_embeds = torch.cat(embeds_list).numpy()
    pickle.dump(open(self.hparams.embeddings_file, 'wb'))

if __name__ == "__main__":
    precompute_embeddings()
