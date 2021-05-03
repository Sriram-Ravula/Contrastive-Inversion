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

from noisy_clip_dataparallel import NoisyCLIP, ContrastiveUnsupervisedDataset, ImageNetCLIPDataset
from linear_probe import LinearProbe

class ImageNetCLIPDatasetTesting(ImageNetCLIPDataset):
    def __init__(self, args):
        super(ImageNetCLIPDatasetTesting,self).__init__(args)

    def test_dataloader(self):
        return self.val_dataloader()

class NoisyCLIPTesting(LightningModule):

    def __init__(self, args, ckpt_file):
        super(NoisyCLIPTesting,self).__init__()
        self.backbone = NoisyCLIP.load_from_checkpoint(ckpt_file).eval()
        self.test_top_1 = Accuracy(top_k=1)
        self.test_top_5 = Accuracy(top_k=5)

    def forward(self, x):
        embed = self.backbone.encode_noisy_image(x)
        return self.backbone(embed)[0]

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = logits.float()
        pred_probs = logits.softmax(dim=-1)

        self.log("test_top_1", self.test_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("test_top_5", self.test_top_5(pred_probs, y), prog_bar=False, logger=False)

    def test_epoch_end(self, outputs):
        self.log("test_top_1", self.test_top_1.compute(), prog_bar=True, logger=True)
        self.log("test_top_5", self.test_top_5.compute(), prog_bar=True, logger=True)
        self.test_top_1.reset()
        self.test_top_5.reset()

def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def zeroshot_eval():
    args = grab_config()
    args.distributed_backend='ddp'
    seed_everything(args.seed)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )

    checkpoint_folder = './Logs_RN101_Adam/'+args.experiment_name+'/checkpoints/'
    checkpoint_file = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])

    trainer = Trainer.from_argparse_args(args, logger=logger)

    dataset = ImageNetCLIPDatasetTesting(args)
    dataset.setup()
    model = NoisyCLIPTesting(args, checkpoint_file)

    trainer.test(model, datamodule=dataset)

if __name__ == "__main__":
    zeroshot_eval()
