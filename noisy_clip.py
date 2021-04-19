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

import torch
import torchvision

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.metrics import Accuracy
from torch.utils.data  import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import CIFAR10

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    """
    This class takes a dataset and creates a contrastive version of that dataset.
    Each item of the dataset is a tuple of a clean image and a noisy image (two
    separate transformations.)
    """
    def __init__(self, clean_dataset, transform_clean=None, transform_noisy=None, return_label=False):
        self.base = clean_dataset
        self.transform_clean = transform_clean
        self.transform_noisy = transform_noisy
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]

        image_clean = self.transform_clean(image_orig) if self.transform_clean is not None else image_orig
        image_noisy = self.transform_noisy(image_orig) if self.transform_noisy is not None else image_orig

        if self.return_label:
            return image_clean, image_noisy, label
        else:
            return image_clean, image_noisy

class CIFAR10CLIPDataset(LightningDataModule):
    """
    Old class for use with CIFAR10. Deprecated since experiments have moved on to ImageNet.
    """
    def __init__(self, train_preprocess, noise_transform, data_dir='./', batch_size=64):
        super(CIFAR10CLIPDataset, self).__init__()
        self.train_preprocess = train_preprocess
        self.noise_transform = noise_transform
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR10(self.data_dir, download=True, train=True)
        CIFAR10(self.data_dir, download=True, train=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset_full = CIFAR10(self.data_dir, download=False, train=True, transform=self.train_preprocess)
            train_data, val_data = random_split(train_dataset_full, [40000,10000])
            self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, self.noise_transform)
            self.val_contrastive = ContrastiveUnsupervisedDataset(val_data, self.noise_transform)

        if stage == 'test' or stage is None:
            self.test_data = CIFAR10(self.data_dir, download=False, train=False, transform=Compose([self.train_preprocess, self.noise_transform]))

    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_contrastive, batch_size=2*self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=2*self.batch_size)


class ImageNetCLIPDataset(LightningDataModule):
    """
    Wrapper class for the ImageNet dataset, handles all data manipulations
    required in order to train the NoisyCLIP model.
    """
    def __init__(self, args):
        super(ImageNetCLIPDataset, self).__init__()

        self.hparams = args
        # self.noise_transform = self.hparams.noise_transform
        self.dataset_dir = self.hparams.dataset_dir
        self.batch_size = self.hparams.batch_size

        # NOTE: training now uses the same distortion as validation (no RandomResizedCrop/RandomHorizontalFlip)
        self.train_set_transform = ImageNetDistortTrain(self.hparams)
        if self.hparams.fixed_mask:
            self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
        else:
            self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):
        train_data = ImageNet100(
        	root=self.hparams.dataset_dir,
            split="train",
            transform=None
        )
        self.val_data = ImageNet100(
            root=self.hparams.dataset_dir,
            split="val",
            transform=self.val_set_transform
        )

        filename = self.hparams.dataset_dir + self.hparams.subset_file_name

        # Get the subset, as well as its labels as text.
        text_labels = list(train_data.idx_to_class.values())

        self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, transform_clean=ImageNetBaseTransform(self.hparams), transform_noisy=self.train_set_transform, return_label=True)

        # Save labels to be reused.
        if self.hparams.save_mapping_and_text:
            pickle.dump(text_labels, open(self.hparams.mapping_and_text_file, 'wb'))

    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2*self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)


class NoisyCLIP(LightningModule):
    def __init__(self, args):
        """
        This model is comprised of two parts: the first is a copy of the baseline
        CLIP model, and the second a copy of its visual encoder. The original image
        encoder is kept frozen, and only its copy is trained.
        """
        super(NoisyCLIP, self).__init__()
        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        if self.hparams.dataset == "Imagenet-100":
            self.N_val = 5000 # Default ImageNet validation set, only 100 classes.
            if self.hparams.mapping_and_text_file is None:
                raise ValueError('No file from which to read text labels was specified.')

            text_labels = pickle.load(open(self.hparams.mapping_and_text_file, 'rb'))
            self.text_list = ['A photo of '+label.strip().replace('_',' ') for label in text_labels]
        else:
            raise NotImplementedError('Handling of the dataset not implemented yet.')

        #self.criterion = None
        #self.criterion = ContrastiveLoss(tau=self.hparams.loss_tau, device=self.hparams.device)
        self.logit_scale = self.hparams.logit_scale
        self.baseclip = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0]
        self.baseclip.eval()
        self.baseclip.requires_grad_(False)
        # self.text_embeddings = self.baseclip.encode_text(clip.tokenize(text_list).to(self.hparams.device))
        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0].visual
        self.noisy_visual_encoder.train()

        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)
        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

    def criterion(self, input1, input2, reduction='mean'):
        bsz = input1.shape[0]
        if self.hparams.loss_type == 'simclr':
            # Create similarity matrix between embeddings.
            full_tensor = torch.cat([input1.unsqueeze(1),input2.unsqueeze(1)], dim=1).view(2*bsz, 1, -1)
            tensor1 = full_tensor.expand(2*bsz,2*bsz,-1)
            tensor2 = full_tensor.permute(1,0,2).expand(2*bsz,2*bsz,-1)
            sim_mat = torch.nn.CosineSimilarity(dim=-1)(tensor1,tensor2)

            # Calculate logits used for the contrastive loss.
            exp_sim_mat = torch.exp(sim_mat/self.hparams.loss_tau)
            mask = torch.ones_like(exp_sim_mat) - torch.eye(2*bsz).to(self.device)
            logmat = -torch.log(exp_sim_mat)+torch.log(torch.sum(mask*exp_sim_mat, 1))

            part1 = torch.sum(torch.diag(logmat, diagonal=1)[np.arange(0,2*bsz,2)])
            part2 = torch.sum(torch.diag(logmat, diagonal=-1)[np.arange(0,2*bsz,2)])
            loss = (part1 + part2)/2
        elif self.hparams.loss_type == 'clip':
            tensor1 = input1 / input1.norm(dim=-1, keepdim=True)
            tensor2 = input2 / input2.norm(dim=-1, keepdim=True)
            sim_mat = (1/self.hparams.loss_tau)*tensor1 @ tensor2.t()
            part1 = F.cross_entropy(sim_mat, torch.LongTensor(np.arange(bsz)).to(self.device))
            part2 = F.cross_entropy(sim_mat.t(), torch.LongTensor(np.arange(bsz)).to(self.device))
            loss = (part1+part2)/2
        elif self.hparams.loss_type == 'mse':
            return F.mse_loss(input2, input1)

        else:
            raise ValueError('Loss function not understood.')

        return loss/bsz if reduction == 'mean' else loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        #sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch: 1/(epoch+1))
        #return [optim], [sched]
        return optim

    def encode_noisy_image(self, image):
        return self.noisy_visual_encoder(image.type(torch.float16))

    def forward(self, image, text=None):
        """
        This forward method is taken from original CLIP model. The use is the same as
        the original function, apart from using the encoder trained for noise images.
        """

        image_features = self.encode_noisy_image(image)
        text_features = self.baseclip.encode_text(clip.tokenize(self.text_list).to(self.device))

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features.type(torch.float16) @ text_features.type(torch.float16).t() # Funny thing, here the original code spells 'iamge' instead of image. Hidden copyright protection? :p
        logits_per_text = self.logit_scale * text_features.type(torch.float16) @ image_features.type(torch.float16).t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    # Training methods
    def training_step(self, train_batch, batch_idx):
        image_clean, image_noisy, labels = train_batch
        embed_clean = self.baseclip.encode_image(image_clean.type(torch.float32))
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = self.criterion(embed_clean, embed_noisy)

        if batch_idx == 0 and self.current_epoch < 20:
            self.logger.experiment.add_image('Train_Sample', img_grid(image_noisy), self.current_epoch)

        image_logits, _ = self.forward(image_noisy)
        image_logits = image_logits.float()
        image_probs = image_logits.softmax(dim=-1)
        # train_top_1 = self.train_top_1(image_probs, labels)
        # train_top_5 = self.train_top_5(image_probs, labels)
        #
        # output = {
        #     'train_loss': loss,
        #     'train_top_1': top_1,
        #     'train_top_5': top_5,
        #     'num_samples': image_clean.shape[0]
        # }
        self.log('train_top_1_step', self.train_top_1(image_probs, labels), prog_bar=False, logger=False)
        self.log('train_top_5_step', self.train_top_5(image_probs, labels), prog_bar=False, logger=False)

        return loss

    # def training_step_end(self, outputs):
    #     num_samples = np.sum([out['num_samples'] for out in outputs])
    #     train_top_1 = np.sum([out['train_top_1'] for out in outputs])
    #     train_top_5 = np.sum([out['train_top_5'] for out in outputs])
    #     train_loss = np.sum([out['train_loss']/out['num_samples'] for out in outputs])/num_samples
    #
    #     full_output = {
    #         'train_loss': train_loss,
    #         'train_top_1': train_top_1,
    #         'train_top_5': train_top_5,
    #         'num_samples': num_samples
    #     }
    #     return full_output

    def training_epoch_end(self, outputs):
        # N_train = np.sum([out['num_samples'] for out in outputs])
        # train_loss = np.sum([out['train_loss']/out['num_samples'] for out in outputs]) / N_train
        # top_1_mean = torch.stack([out['train_top_1'] for out in outputs]).sum() / N_train
        # top_5_mean = torch.stack([out['train_top_5'] for out in outputs]).sum() / N_train
        # self.log("train_loss", top_1_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.log("train_top_1", top_1_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.log("train_top_5", top_5_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_top_1', self.train_top_1.compute(), prog_bar=True, logger=True)
        self.log('train_top_5', self.train_top_5.compute(), prog_bar=True, logger=True)

    # Validation methods
    def validation_step(self, test_batch, batch_idx):
        images_noisy, labels = test_batch
        image_logits, _ = self.forward(images_noisy)

        if batch_idx == 0 and self.current_epoch < 20:
            self.logger.experiment.add_image('Val_Sample', img_grid(images_noisy), self.current_epoch)

        image_logits, _ = self.forward(images_noisy)
        image_logits = image_logits.float()
        image_probs = image_logits.softmax(dim=-1)
        self.log('val_top_1_step', self.val_top_1(image_probs, labels), prog_bar=False, logger=False)
        self.log('val_top_5_step', self.val_top_5(image_probs, labels), prog_bar=False, logger=False)

        # loss = torch.nn.CrossEntropyLoss()(image_logits, labels)
        # top_1 = top_k_accuracy(image_logits, labels, k=1)
        # top_5 = top_k_accuracy(image_logits, labels, k=5)
        #
        # output = {
        #     'val_loss': loss,
        #     'val_top_1': top_1,
        #     'val_top_5': top_5
        # }
        #
        # return output

    # def validation_step_end(self, outputs):
    #     num_samples = np.sum([out['num_samples'] for out in outputs])
    #     val_top_1 = np.sum([out['val_top_1'] for out in outputs])
    #     val_top_5 = np.sum([out['val_top_5'] for out in outputs])
    #     val_loss = np.sum([out['val_loss']/out['num_samples'] for out in outputs])/num_samples
    #
    #     full_output = {
    #         'val_loss': val_loss,
    #         'val_top_1': val_top_1,
    #         'val_top_5': val_top_5,
    #         'num_samples': num_samples
    #     }
    #     return full_output

    def validation_epoch_end(self, outputs):
        # N_val = np.sum([out['num_samples'] for out in outputs])
        # val_loss = np.sum([out['val_loss']/out['num_samples'] for out in outputs]) / N_val
        # top_1_mean = torch.stack([out['val_top_1'] for out in outputs]).sum() / N_val
        # top_5_mean = torch.stack([out['val_top_5'] for out in outputs]).sum() / N_val
        #
        # # Debug assertion, if not then something went real bad somewhere
        # assert(N_val == self.N_val)
        #
        # self.log("val_loss", 1 - top_5_mean, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True) #VAL_LOSS IS ACTUALLY (1 - TOP_5) FOR CHECKPOINTING
        # self.log("top_1", top_1_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.log("top_5", top_5_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.log("val_ce_loss", val_loss_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_top_1', self.val_top_1.compute(), prog_bar=True, logger=True)
        self.log('val_top_5', self.val_top_5.compute(), prog_bar=True, logger=True)

def run_noisy_clip():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    #args.plugins = DDPPlugin(find_unused_parameters=False)

    seed_everything(args.seed)

    dataset = ImageNetCLIPDataset(args)
    dataset.setup()
    model = NoisyCLIP(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer.logger = logger

    trainer.fit(model, dataset)


if __name__ == "__main__":
    run_noisy_clip()
