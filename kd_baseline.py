#!/usr/bin/env python

import sys
import os
import argparse
import torch
from torch import Tensor
import torch.nn.functional as F
from clip_files import model, clip

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data  import random_split, DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from baselines import Baseline

class ImageNetCLIPDataset(LightningDataModule):
    """
    Want this to return (clean img, distorted img, class#)
    """
    def __init__(self, args):
        super(ImageNetCLIPDataset, self).__init__()

        self.hparams = args

        self.dataset_dir = self.hparams.dataset_dir
        self.batch_size = self.hparams.batch_size

        if self.hparams.distortion == "None":
            self.train_set_transform = ImageNetBaseTrainContrastive(self.hparams)
            self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
        elif self.hparams.distortion == 'multi':
            self.train_set_transform = ImageNetDistortTrainMultiContrastive(self.hparams)
            self.val_set_transform = ImageNetDistortValMulti(self.hparams)
        else:
            #set up the training transform and if we want a fixed mask, transfer the same mask to the validation transform
            self.train_set_transform = ImageNetDistortTrainContrastive(self.hparams)

            if self.hparams.fixed_mask:
                self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
            else:
                self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):
        train_data = ImageNet100(
        	root=self.hparams.dataset_dir,
            split="train",
            transform=self.train_set_transform
        )
        val_data = ImageNet100(
            root=self.hparams.dataset_dir,
            split="val",
            transform=self.val_set_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)

class KDBaseline(LightningModule):
    def __init__(self, args):
        super(KDBaseline, self).__init__()
        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        #(1) Load the correct dataset class names
        if self.hparams.dataset == "ImageNet100" or self.hparams.dataset == "Imagenet-100":
            self.N_val = 5000 # Default ImageNet validation set, only 100 classes.
        else:
            raise NotImplementedError('Handling of the dataset not implemented yet.')

        #(2) set up the teacher RN network - freeze it and don't use gradients!
        #Grab the correct Resnet model
        self.teacher = Baseline.load_from_checkpoint(args.checkpoint_path)
        self.teacher = self.teacher.encoder
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        #(3) set up the student CLIP network - unfreeze it and use gradients!
        self.student = Baseline.load_from_checkpoint(args.checkpoint_path)
        self.student = self.student.encoder
        self.student.train()
        self.student.requires_grad_(True)

        #Set up losses and stuff
        label_loss = nn.CrossEntropyLoss(reduction = "mean")
        

    def criterion(self, teacher_prediction, student_prediction, label):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = self.hparams.alpha
        T = self.hparams.temperature

        KD_loss = nn.KLDivLoss()(F.log_softmax(student_prediction/T, dim=1),
                                F.softmax(teacher_prediction/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(student_prediction, labels) * (1. - alpha)

        return KD_loss


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.student.parameters(), lr = self.hparams.lr)

        num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    def encode_noisy_image(self, image):
        """
        Return S(yi) where S() is the student network and yi is distorted images.
        """

        return self.student(image)

    def forward(self, image):

        return self.student(image)

    # Training methods - here we are concerned with contrastive loss (or MSE) between clean and noisy image embeddings.
    def training_step(self, train_batch, batch_idx):
        """
        Takes a batch of clean and noisy images and returns their respective embeddings.

        Returns:
            embed_clean: T(xi) where T() is the teacher and xi are clean images. Shape [N, embed_dim]
            embed_noisy: S(yi) where S() is the student and yi are noisy images. Shape [N, embed_dim]
        """
        image_clean, image_noisy = train_batch

        self.teacher.eval()
        with torch.no_grad():
            embed_clean = self.teacher(image_clean).flatten(1)

        embed_noisy = self.encode_noisy_image(image_noisy).flatten(1)

        return {'embed_clean': embed_clean, 'embed_noisy': embed_noisy}

    def training_step_end(self, outputs):
        """
        Given all the clean and noisy image embeddings form across GPUs from training_step, gather them onto a single GPU and calculate overall loss.
        """
        embed_clean_full = outputs['embed_clean']
        embed_noisy_full = outputs['embed_noisy']

        loss = self.criterion(embed_clean_full, embed_noisy_full)

        self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True, on_step=True, on_epoch=True)

        return loss

    # Validation methods - here we are concerned with similarity between noisy image embeddings and classification text embeddings.
    def validation_step(self, val_batch, batch_idx):
        """
        Grab the noisy image embeddings: S(yi), where S() is the student and yi = Distort(xi). Done on each GPU.
        Return these to be evaluated in validation step end.
        """
        image_clean, image_noisy = val_batch

        with torch.no_grad():
            embed_clean = self.teacher(image_clean).flatten(1)

            embed_noisy = self.encode_noisy_image(image_noisy).flatten(1)

        return {'embed_clean': embed_clean, 'embed_noisy': embed_noisy}

    def validation_step_end(self, outputs):
        """
        Gather the noisy image features and their labels from each GPU.
        Then calculate their similarities, convert to probabilities, and calculate accuracy on each GPU.
        """
        embed_clean_full = outputs['embed_clean']
        embed_noisy_full = outputs['embed_noisy']

        loss = self.criterion(embed_clean_full, embed_noisy_full)

        self.log('val_simclr_loss', loss, prog_bar=True, logger=True, sync_dist=True, on_step=True, on_epoch=True)

def run_noisy_student():
    args = grab_config()

    seed_everything(args.seed)

    dataset = ImageNetCLIPDataset(args)
    dataset.setup()
    model = KDBaseline(args)

    logger = TensorBoardLogger(
        save_dir= args.logdir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[ModelCheckpoint(save_top_k=-1, period=25)])      

    trainer.fit(model, dataset)

def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    run_noisy_student()
