#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from torch.utils.data  import DataLoader

from noisy_clip_birds import NoisyCLIPBirds
from u2nets import BGMask
import spur_datasets.waterbirds as wb

class LinearProbeBirds(LightningModule):
    """
    A class to train and evaluate a linear probe on top of representations learned from a noisy clip student.
    """
    def __init__(self, args):
        super(LinearProbeBirds, self).__init__()
        self.hparams = args

        #(1) Set up the dataset
        #Here, we use a 100-class subset of ImageNet

        #This should be initialised as a trained student CLIP network
        saved_student = NoisyCLIPBirds.load_from_checkpoint(self.hparams.checkpoint_path)

        self.backbone = saved_student.noisy_visual_encoder
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
            noisy_embeddings = self.backbone(x.type(torch.float16))

        return self.output(noisy_embeddings.float())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.output.parameters(), lr = self.hparams.lr)

        if self.hparams.dataset == "ImageNet100":
            num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters
            if self.hparams.use_subset:
                num_steps = num_steps * self.hparams.subset_ratio
        else:
            num_steps = 500

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

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

    model = LinearProbeBirds(args)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger)

    train_dataset = BirdModule(root_dir=args.dataset_dir, groups=[0,3], batch_size=args.batch_size)
    test_dataset_1 = BirdModule(root_dir=args.dataset_dir, groups=1, batch_size=args.batch_size)
    test_dataset_2 = BirdModule(root_dir=args.dataset_dir, groups=1, batch_size=args.batch_size)

    trainer.fit(model, datamodule=train_dataset)
    trainet.test(model, datamodule=test_dataset_1)
    trainet.test(model, datamodule=test_dataset_2)

if __name__ == "__main__":
    linear_eval()
