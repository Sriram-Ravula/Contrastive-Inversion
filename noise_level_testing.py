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

from linear_probe import LinearProbe
from baselines import Baseline
from zeroshot_validation import NoisyCLIPTesting

class ImageNet100Test(LightningDataModule):
    """
    This class loads the validation set of ImageNet100, to be used only for testing.
    Implemented separately because it is linked to both end-to-end suprevised and contrastive training.
    """
    def __init__(self, args):
        super(ImageNet100Test, self).__init__()

        self.hparams = args

        self.dataset_dir = self.hparams.dataset_dir

        if self.hparams.distortion == "None":
            self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
        else:
            self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):
        self.val_data = ImageNet100(
            root=self.hparams.dataset_dir,
            split="val",
            transform=self.val_set_transform
        )

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=512, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)


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
   
    for noise_level in args.noise_levels:
        #Choose the appropriate model based on type, and load from checkpoint.
        trainer = Trainer.from_argparse_args(args, logger=logger)
        if args.saved_model_type == 'linear':
            saved_model = LinearProbe.load_from_checkpoint(args.checkpoint_path)
        elif args.saved_model_type == 'baseline':
            saved_model = Baseline.load_from_checkpoint(args.checkpoint_path)
        elif args.saved_model_type == 'zeroshot':
            saved_model = NoisyCLIPTesting(args, args.checkpoint_path)
            
        # Correctly define noise levels to test.
        if args.distortion == "squaremask":
            args.length = noise_level
        elif args.distortion == "randommask":
            args.percent_missing = noise_level
        elif args.distortion == "gaussiannoise":
            args.std = noise_level
        elif args.distortion == "gaussianblur":
            args.kernel_size = noise_level[0]
            args.sigma = noise_level[1]

        test_data = ImageNet100Test(args)
        trainer.test(model=saved_model, datamodule=test_data, verbose=True)

if __name__ == "__main__":
    noise_level_eval()
