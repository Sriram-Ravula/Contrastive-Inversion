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
        self.train_data = ImageNet100(
        	root=self.hparams.dataset_dir,
            split="train",
            transform=self.train_set_transform
        )
        self.val_data = ImageNet100(
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
        teacher = Baseline.load_from_checkpoint(args.checkpoint_path)
        self.teacher = teacher.encoder
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        #(3) set up the student CLIP network - unfreeze it and use gradients!
        student = Baseline.load_from_checkpoint(args.checkpoint_path)
        self.student = student.encoder
        self.student.train()
        self.student.requires_grad_(True)

        #Set up losses and stuff
        self.supervised_loss = nn.CrossEntropyLoss(reduction = "sum")
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')


        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)

        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

        self.test_top_1 = Accuracy(top_k=1)
        self.test_top_5 = Accuracy(top_k=5)
        

    def criterion(self, teacher_prediction, student_prediction, label):
        """
        Compute the knowledge-distillation (KD) loss given outputs as logits, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = self.hparams.alpha
        T = self.hparams.temperature

        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_prediction/T, dim=1),
                                F.softmax(teacher_prediction/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(student_prediction, label) * (1. - alpha)

        return KD_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.student.parameters(), lr = self.hparams.lr)

        num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    def forward(self, image):

        return self.student(image)

    # Training methods - here we are concerned with contrastive loss (or MSE) between clean and noisy image embeddings.
    def training_step(self, train_batch, batch_idx):
        image_clean, image_noisy = train_batch[0]
        label = train_batch[1]

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Train_Sample_Noisy', img_grid(image_noisy), self.current_epoch)
            self.logger.experiment.add_image('Train_Sample_Clean', img_grid(image_clean), self.current_epoch)

        self.teacher.eval()
        with torch.no_grad():
            embed_clean = self.teacher(image_clean).flatten(1)

        embed_noisy = self.forward(image_noisy).flatten(1)

        pred_probs = embed_noisy.softmax(dim=-1)

        loss = self.criterion(embed_clean, embed_noisy, label)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, sync_dist_op='sum')
        self.log("train_top_1", self.train_top_1(pred_probs, label), prog_bar=False, logger=False)
        self.log("train_top_5", self.train_top_5(pred_probs, label), prog_bar=False, logger=False)

        return loss

    def training_epoch_end(self, outputs):
        self.log("train_top_1", self.train_top_1.compute(), prog_bar=True, logger=True)
        self.log("train_top_5", self.train_top_5.compute(), prog_bar=True, logger=True)

        self.train_top_1.reset()
        self.train_top_5.reset()

    # Validation methods - here we are concerned with similarity between noisy image embeddings and classification text embeddings.
    def validation_step(self, val_batch, batch_idx):
        image_noisy, label = val_batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Val_Sample_Noisy', img_grid(image_noisy), self.current_epoch)

        embed_noisy = self.forward(image_noisy).flatten(1)

        pred_probs = embed_noisy.softmax(dim=-1)

        self.log("val_top_1", self.val_top_1(pred_probs, label), prog_bar=False, logger=False)
        self.log("val_top_5", self.val_top_5(pred_probs, label), prog_bar=False, logger=False)
    
    def validation_epoch_end(self, outputs):
        self.log("val_top_1", self.val_top_1.compute(), prog_bar=True, logger=True)
        self.log("val_top_5", self.val_top_5.compute(), prog_bar=True, logger=True)

        self.val_top_1.reset()
        self.val_top_5.reset()

    def test_step(self, test_batch, batch_idx):
        image_noisy, label = test_batch

        embed_noisy = self.forward(image_noisy).flatten(1)

        pred_probs = embed_noisy.softmax(dim=-1)

        self.log("test_top_1", self.test_top_1(pred_probs, label), prog_bar=False, logger=False)
        self.log("test_top_5", self.test_top_5(pred_probs, label), prog_bar=False, logger=False)
    
    def test_epoch_end(self, outputs):
        self.log("test_top_1", self.test_top_1.compute(), prog_bar=True, logger=True)
        self.log("test_top_5", self.test_top_5.compute(), prog_bar=True, logger=True)

        self.test_top_1.reset()
        self.test_top_5.reset()


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
