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

import torch.nn as nn
import torch
import torchvision

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.metrics import Accuracy
from torch.utils.data  import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class LinearProbe(LightningModule):
    def __init__(self, n_classes, emb_dim, backbone):
        self.n_classes = n_classes
        self.emb_dim = emb_dim

        self.backbone = backbone

        self.output = nn.Linear(self.emb_dim, self.n_classes)

        self.criterion = nn.CrossEntropyLoss(reduction = "sum")
    
    def forward(self, x):
        """
        Given a set of image embeddings with shape [N, emb_dim], output logits
        """
        return self.output(x)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.output.parameters(), lr = self.lr)

        num_steps = 5000//(self.hparams.batch_size * self.hparams.gpus)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]
    
    
