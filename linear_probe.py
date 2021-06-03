#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn

from clip_files import clip
from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from torch.utils.data  import DataLoader

from noisy_clip_dataparallel import NoisyCLIP

class LinearProbe(LightningModule):
    """
    A class to train and evaluate a linear probe on top of representations learned from a noisy clip student.
    """
    def __init__(self, args):
        super(LinearProbe, self).__init__()
        self.hparams = args

        #(1) Set up the dataset
        #Here, we use a 100-class subset of ImageNet
        if self.hparams.dataset not in ["ImageNet100", "CIFAR10", "CIFAR100"]:
            raise ValueError("Unsupported dataset selected.")
        else:
            if self.hparams.distortion == "None":
                self.train_set_transform = ImageNetBaseTransform(self.hparams)
                self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
            elif self.hparams.distortion == "multi":
                self.train_set_transform = ImageNetDistortTrainMulti(self.hparams)
                self.val_set_transform = ImageNetDistortValMulti(self.hparams)

            else:
                #If we are using the ImageNet dataset, then set up the train and val sets to use the same mask if needed!
                self.train_set_transform = ImageNetDistortTrain(self.hparams)

                if self.hparams.fixed_mask:
                    self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
                else:
                    self.val_set_transform = ImageNetDistortVal(self.hparams)

        #This should be initialised as a trained student CLIP network
        if self.hparams.encoder == "clip":
            saved_student = NoisyCLIP.load_from_checkpoint(self.hparams.checkpoint_path)
            self.backbone = saved_student.noisy_visual_encoder
        elif self.hparams.encoder == "clean":
            saved_student = clip.load('RN101', 'cpu', jit=False)[0]
            self.backbone = saved_student.visual


        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        #This is the meat
        self.output = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

        #Set up training and validation metrics
        self.criterion = nn.CrossEntropyLoss()

        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)
        self.test_top_1 = Accuracy(top_k=1)
        self.test_top_5 = Accuracy(top_k=5)


    def forward(self, x):
        """
        Given a set of images x with shape [N, c, h, w], get their embeddings and then logits.

        Returns: Logits with shape [N, n_classes]
        """

        #Grab the noisy image embeddings
        self.backbone.eval()
        with torch.no_grad():
            if self.hparams.encoder == "clip":
                noisy_embeddings = self.backbone(x.type(torch.float16)).float()
            elif self.hparams.encoder == "clean":
                noisy_embeddings = self.backbone(x.type(torch.float16)).float()

        return self.output(noisy_embeddings)

    def configure_optimizers(self):
        if not hasattr(self.hparams, 'weight_decay'):
            self.hparams.weight_decay = 0

        opt = torch.optim.Adam(self.output.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)

        if self.hparams.dataset == "ImageNet100":
            num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters
            if self.hparams.use_subset:
                num_steps = num_steps * self.hparams.subset_ratio
        else:
            num_steps = 500

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    def train_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            train_dataset = ImageNet100(
                root=self.hparams.dataset_dir,
                split = 'train',
                transform = self.train_set_transform
            )
        N_train = len(train_dataset)
        if self.hparams.use_subset:
            train_dataset = few_shot_dataset(train_dataset, int(np.ceil(N_train*self.hparams.subset_ratio/self.hparams.num_classes)))

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

        val_dataloader = DataLoader(val_dataset, batch_size=4*self.hparams.batch_size, num_workers=self.hparams.workers,\
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
        self.val_top_1.reset()
        self.val_top_5.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        self.log("test_top_1", self.test_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("test_top_5", self.test_top_5(pred_probs, y), prog_bar=False, logger=False)

    def test_epoch_end(self, outputs):
        self.log("test_top_1", self.test_top_1.compute(), prog_bar=True, logger=True)
        self.log("test_top_5", self.test_top_5.compute(), prog_bar=True, logger=True)
        self.test_top_1.reset()
        self.test_top_5.reset()

    def predict(self, batch, batch_idx, hiddens):
        x, y = batch
        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)
        preds = pred_probs.argmax(dim=-1)
        return preds


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
