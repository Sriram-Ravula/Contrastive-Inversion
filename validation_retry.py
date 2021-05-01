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


def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def noise_level_eval():
    args = grab_config()

    seed_everything(args.seed)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )

    checkpoint_folder = './Logs_RN101_Adam/'+args.experiment_name+'/checkpoints/'
    checkpoint_file = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])

    trainer = Trainer.from_argparse_args(args, logger=logger)

    dataset = ImageNetCLIPDataset(args)
    dataset.setup()
    saved_model = NoisyCLIP.load_from_checkpoint(checkpoint_file)
    saved_model.val_top_1.reset()
    saved_model.val_top_5.reset()

    trainer.validate(saved_model, dataset, verbose=True)

if __name__ == "__main__":
    noise_level_eval()
