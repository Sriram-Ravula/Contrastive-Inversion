import sys
import os
import argparse
import torch
from torch import Tensor
import torch.nn.functional as F
from clip_files import model, clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, STL10
import torchvision.models as models

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.metrics import Accuracy, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint

from noisy_clip_dataparallel import NoisyCLIP
from baselines import Baseline

def grab_transforms(args):
    """
    Method to get the correct train and val set transforms given configuration parameters

    Args:
        args - arguments from a config file

    Returns:
        train_set_transform -
        val_set_transform -
    """
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100" or args.dataset == "STL10":
        #Get the correct image transform
        if args.distortion == "None":
            train_set_transform = GeneralBaseTransform(args)
            val_set_transform = GeneralBaseTransformVal(args)
        elif args.distortion == "multi":
            train_set_transform = GeneralDistortTrainMulti(args)
            val_set_transform = GeneralDistortValMulti(args)
        else:
            train_set_transform = GeneralDistortTrain(args)
            val_set_transform = GeneralDistortVal(args)

    elif args.dataset == 'COVID' or args.dataset == 'ImageNet100B' or args.dataset == 'imagenet-100B':
        #Get the correct image transform
        if args.distortion == "None":
            train_set_transform = ImageNetBaseTransform(args)
            val_set_transform = ImageNetBaseTransformVal(args)
        elif args.distortion == "multi":
            train_set_transform = ImageNetDistortTrainMulti(args)
            val_set_transform = ImageNetDistortValMulti(args)
        else:
            train_set_transform = ImageNetDistortTrain(args)
            val_set_transform = ImageNetDistortVal(args)

    return train_set_transform, val_set_transform


class TransferLearning(LightningModule):
    def __init__(self, args):
        super(TransferLearning, self).__init__()

        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        self.train_set_transform, self.val_set_transform = grab_transforms(self.hparams)

        #Grab the correct model - only want the embeddings from the final layer!
        if self.hparams.saved_model_type == 'contrastive':
            saved_model = NoisyCLIP.load_from_checkpoint(self.hparams.checkpoint_path)
            self.backbone = saved_model.noisy_visual_encoder
        elif self.hparams.saved_model_type == 'baseline':
            saved_model = Baseline.load_from_checkpoint(self.hparams.checkpoint_path)
            self.backbone = saved_model.encoder.feature_extractor
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        #Set up a classifier with the correct dimensions
        self.output = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)

        #Set up the criterion and stuff
        #(3) Set up our criterion - here we use reduction as "sum" so that we are able to average over all validation sets
        self.criterion = nn.CrossEntropyLoss(reduction = "mean")

        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)

        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

        self.test_top_1 = Accuracy(top_k=1)
        self.test_top_5 = Accuracy(top_k=5)

        #class INFECTED has label 0
        if self.hparams.dataset == 'COVID':
            self.val_auc = AUROC(pos_label=0)

            self.test_auc = AUROC(pos_label=0)

    def forward(self, x):
        #Grab the noisy image embeddings
        self.backbone.eval()
        with torch.no_grad():
            if self.hparams.encoder == "clip":
                noisy_embeddings = self.backbone(x.type(torch.float16)).float()
            elif self.hparams.encoder == "resnet":
                noisy_embeddings = self.backbone(x)

        return self.output(noisy_embeddings.flatten(1))

    def configure_optimizers(self):
        if not hasattr(self.hparams, 'weight_decay'):
            self.hparams.weight_decay = 0

        opt = torch.optim.Adam(self.output.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)

        num_steps = self.hparams.max_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    def _grab_dataset(self, split):
        """
        Given a split ("train" or "val" or "test") and a dataset, returns the proper dataset.
        Dataset needed is defined in this object's hparams

        Args:
            split: the split to use in the dataset
        Returns:
            dataset: the desired dataset with the correct split
        """
        if self.hparams.dataset == "CIFAR10":
            if split == 'train':
                train = True
                transform = self.train_set_transform
            else:
                train = False
                transform = self.val_set_transform

            dataset = CIFAR10(root=self.hparams.dataset_dir, train=train, transform=transform, download=True)

        elif self.hparams.dataset == "CIFAR100":
            if split == 'train':
                train = True
                transform = self.train_set_transform
            else:
                train = False
                transform = self.val_set_transform

            dataset = CIFAR100(root=self.hparams.dataset_dir, train=train, transform=transform, download=True)

        elif self.hparams.dataset == 'STL10':
            if split == 'train':
                stlsplit = 'train'
                transform = self.train_set_transform
            else:
                stlsplit = 'test'
                transform = self.val_set_transform

            dataset = STL10(root=self.hparams.dataset_dir, split=stlsplit, transform = transform, download=True)

        elif self.hparams.dataset == 'COVID':
            if split == 'train':
                covidsplit = 'train'
                transform = self.train_set_transform
            else:
                covidsplit = 'test'
                transform = self.val_set_transform

            dataset = torchvision.datasets.ImageFolder(root = self.hparams.dataset_dir + covidsplit, transform=transform)

        elif self.hparams.dataset == 'ImageNet100B' or self.hparams.dataset == 'imagenet-100B':
            if split == 'train':
                transform = self.train_set_transform
            else:
                split = 'val'
                transform = self.val_set_transform

            dataset = ImageNet100(root = self.hparams.dataset_dir, split=split, transform=transform)

        elif self.hparams.dataset == 'COVID':
            if split == 'train':
                covidsplit = 'train'
                transform = self.train_set_transform
            else:
                covidsplit = 'test'
                transform = self.val_set_transform

            dataset = torchvision.datasets.ImageFolder(root = self.hparams.dataset_dir + covidsplit, transform=transform)

        elif self.hparams.dataset == 'ImageNet100B' or self.hparams.dataset == 'imagenet-100B':
            if split == 'train':
                transform = self.train_set_transform
            else:
                split = 'val'
                transform = self.val_set_transform

            dataset = ImageNet100(root = self.hparams.dataset_dir, split=split, transform=transform)

        return dataset


    def train_dataloader(self):
        train_dataset = self._grab_dataset(split='train')

        N_train = len(train_dataset)
        if self.hparams.use_subset:
            train_dataset = few_shot_dataset(train_dataset, int(np.ceil(N_train*self.hparams.subset_ratio/self.hparams.num_classes)))

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        val_dataset = self._grab_dataset(split='val')

        N_val = len(val_dataset)

        #SET SHUFFLE TO TRUE SINCE AUROC FREAKS OUT IF IT GETS AN ALL-1 OR ALL-0 BATCH
        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = self._grab_dataset(split='test')

        N_test = len(test_dataset)

        #SET SHUFFLE TO TRUE SINCE AUROC FREAKS OUT IF IT GETS AN ALL-1 OR ALL-0 BATCH
        test_dataloader = DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return test_dataloader

    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Train_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=False, on_step=True, \
                    on_epoch=True, logger=True, sync_dist=True, sync_dist_op='sum')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Val_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1) #(N, num_classes)

        if self.hparams.dataset == 'COVID':
            positive_prob = pred_probs[:, 0].flatten() #class 0 is INFECTED label
            true_labels = y.flatten()

            self.val_auc.update(positive_prob, true_labels)

            self.log("val_auc", self.val_auc, prog_bar=False, logger=False)

        self.log("val_top_1", self.val_top_1(pred_probs, y), prog_bar=False, logger=False)

        if self.hparams.dataset != 'COVID':
            self.log("val_top_5", self.val_top_5(pred_probs, y), prog_bar=False, logger=False)

    def validation_epoch_end(self, outputs):
        self.log("val_top_1", self.val_top_1.compute(), prog_bar=True, logger=True)

        if self.hparams.dataset != 'COVID':
            self.log("val_top_5", self.val_top_5.compute(), prog_bar=True, logger=True)

        if self.hparams.dataset == 'COVID':
            self.log("val_auc", self.val_auc.compute(), prog_bar=True, logger=True)

            self.val_auc.reset()

        self.val_top_1.reset()
        self.val_top_5.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        if self.hparams.dataset == 'COVID':
            positive_prob = pred_probs[:, 0].flatten() #class 0 is INFECTED label
            true_labels = y.flatten()

            self.test_auc.update(positive_prob, true_labels)

            self.log("test_auc", self.test_auc, prog_bar=False, logger=False)

        self.log("test_top_1", self.test_top_1(pred_probs, y), prog_bar=False, logger=False)
        if self.hparams.dataset != 'COVID':
            self.log("test_top_5", self.test_top_5(pred_probs, y), prog_bar=False, logger=False)

    def test_epoch_end(self, outputs):
        self.log("test_top_1", self.test_top_1.compute(), prog_bar=True, logger=True)
        if self.hparams.dataset != 'COVID':
            self.log("test_top_5", self.test_top_5.compute(), prog_bar=True, logger=True)

        if self.hparams.dataset == 'COVID':
            self.log("test_auc", self.test_auc.compute(), prog_bar=True, logger=True)

            self.test_auc.reset()

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

def transfer_learning():
    args = grab_config()

    seed_everything(args.seed)

    model = TransferLearning(args)

    logger = TensorBoardLogger(
        save_dir= args.logdir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger)

    trainer.fit(model)

if __name__ == '__main__':
    transfer_learning()
