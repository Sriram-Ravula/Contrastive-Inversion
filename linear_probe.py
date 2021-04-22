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

from noisy_clip_dataparallel import NoisyCLIP

class LinearProbe(LightningModule):
    def __init__(self, args, backbone):
        self.hparams = args

        #(1) Set up the dataset
        #Here, we use a 100-class subset of ImageNet
        if self.hparams.dataset != "ImageNet100":
            raise ValueError("Unsupported dataset selected.")
        else:
            if self.hparams.distortion == "None":
                self.train_set_transform = ImageNetBaseTransform(self.hparams)
                self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
            else:
                #If we are using the ImageNet dataset, then set up the train and val sets to use the same mask if needed! 
                self.train_set_transform = ImageNetDistortTrain(self.hparams)
            
                if self.hparams.fixed_mask:        
                    self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
                else:
                    self.val_set_transform = ImageNetDistortVal(self.hparams)

        #This should be initialised as a trained student CLIP network
        saved_student = NoisyCLIP.load_from_checkpoint(self.hparams.checkpoint_path)

        self.backbone = saved_student.noisy_visual_encoder
        self.backbone.eval() 
        for param in self.backbone.parameters():
            param.requires_grad = False

        #This is the meat
        self.output = nn.Linear(self.hparams.emb_dim, self.hparams.n_classes)

        #Set up training and validation metrics
        self.criterion = nn.CrossEntropyLoss(reduction = "sum")

        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)
    
    def forward(self, x):
        """
        Given a set of images x with shape [N, c, h, w], get their embeddings and then logits.

        Returns: Logits with shape [N, n_classes]
        """

        #Grab the noisy image embeddings
        self.backbone.eval()
        with torch.no_grad():
            noisy_embeddings = self.backbone(x.type(torch.float16))

        return self.output(noisy_embeddings.float())
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.output.parameters(), lr = self.lr)

        num_steps = 5000//(self.hparams.batch_size * self.hparams.gpus)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]
    
    def train_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            train_dataset = ImageNet100(
                root=self.hparams.dataset_dir,
                split = 'train',
                transform = self.train_set_transform
            )

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            val_dataset = ImageNet100(
                root=self.hparams.dataset_dir,
                split = 'val',
                transform = self.val_set_transform
            )

            self.N_val = 5000

        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)

        return val_dataloader

    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Train_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, \
                    on_epoch=True, logger=True, sync_dist=True, sync_dist_op='sum')

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Val_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        self.log("val_top_1", self.val_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("val_top_5", self.val_top_5(pred_probs, y), prog_bar=False, logger=False)
    
    def validation_epoch_end(self, outputs):
        self.log("val_top_1", self.val_top_1.compute(), prog_bar=True, logger=True)
        self.log("val_top_5", self.val_top_5.compute(), prog_bar=True, logger=True)

def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def linear_eval():
    args = grab_config()

    seed_everything(args.seed)

    model = LinearProbe(args)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger)

    trainer.fit(model)

if __name__ == "__main__":
    linear_eval()
