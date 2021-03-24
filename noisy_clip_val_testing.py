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
from torch.utils.data  import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import CIFAR10

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    """
    This class takes a dataset and creates a contrastive version of that dataset.
    Each item of the dataset is a tuple of a clean image and a noisy image (two
    separate transformations.)
    """
    def __init__(self, clean_dataset, transform_clean=None, transform_noisy=None, return_label_for_val=False):
        self.base = clean_dataset
        self.transform_clean = transform_clean
        self.transform_noisy = transform_noisy
        self.return_label_for_val = return_label_for_val

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]

        image_clean = self.transform_clean(image_orig) if self.transform_clean is not None else image_orig
        image_noisy = self.transform_noisy(image_orig) if self.transform_noisy is not None else image_orig

        if self.return_label_for_val:
            return image_clean, image_noisy, labels
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
        self.train_set_transform = ImageNetDistortTrain(self.hparams)
        self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):
        train_data = torchvision.datasets.ImageNet(
        	root=self.hparams.dataset_dir,
            split="train",
            transform=None
        )
        val_data = torchvision.datasets.ImageNet(
            root=self.hparams.dataset_dir,
            split="val",
            transform=None
        )

        filename = self.hparams.dataset_dir + self.hparams.subset_file_name

        # Get the subset, as well as its labels as text.
        train_data, og_to_new_dict, text_labels = get_subset(train_data, filename=filename, return_class_labels=True)
        val_data, _ = get_subset(val_data, filename=filename)

        self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, transform_clean=ImageNetBaseTransform(self.hparams), transform_noisy=self.train_set_transform)
        self.val_contrastive = ContrastiveUnsupervisedDataset(val_data, transform_clean=ImageNetBaseTransformVal(self.hparams), transform_noisy=self.val_set_transform, return_label_for_val=True)
        # Save mapping/labels to be reused.
        if self.hparams.save_mapping_and_text:
            pickle.dump((og_to_new_dict, text_labels), open(self.hparams.mapping_and_text_file, 'wb'))

    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)


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
                raise ValueError('No file from which to read mapping/text labels was specified.')

            og_to_new_dict, text_labels = pickle.load(open(self.hparams.mapping_and_text_file, 'rb'))
            self.class_map = og_to_new_dict
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

        elif self.hparams.loss_type == 'mse':
            return F.mse_loss(input1, input2)

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

    def training_step(self, train_batch, batch_idx):
        #if self.criterion is None:
        #    if self.hparams.loss == 'contrastive':
        #        self.criterion = ContrastiveLoss(self.hparams.tau, self.device)
        #        self.baseclip = clip.load(self.hparams.baseclip_type, self.device, jit=False)[0]
        #        self.baseclip.eval()
        #        self.text_embeddings = self.baseclip.encode_text(clip.tokenize(text_list).to(self.device))
        #        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.device, jit=False)[0].visual
        #        self.noisy_visual_encoder.train()
        #    else:
        #        raise NotImplementedError('Loss function not implemented yet.')

        image_clean, image_noisy = train_batch
        embed_clean = self.baseclip.encode_image(image_clean.type(torch.float32))
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = self.criterion(embed_clean, embed_noisy)

        loss_dict = {"Train_Loss": loss}

        output = {
            'loss': loss,
            'progress_bar': loss_dict,
            'log': loss_dict
        }

        return output

    def validation_step(self, test_batch, batch_idx):
        #if self.criterion is None:
        #    if self.hparams.loss == 'contrastive':
        #        self.criterion = ContrastiveLoss(self.hparams.tau, self.device)
        #        self.baseclip = clip.load(self.hparams.baseclip_type, self.device, jit=False)[0]
        #        self.baseclip.eval()
        #        self.text_embeddings = self.baseclip.encode_text(clip.tokenize(text_list).to(self.device))
        #        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.device, jit=False)[0].visual
        #        self.noisy_visual_encoder.train()
        #    else:
        #        raise NotImplementedError('Loss function not implemented yet.')


        images_clean, images_noisy, labels = test_batch
        image_logits, _ = self.forward(images_noisy)
        preds = torch.argmax(image_logits.softmax(dim=-1), axis=1)

        if self.hparams.dataset == "Imagenet-100":
            labels = map_classes(labels, self.class_map)

        if batch_idx == 0 and self.current_epoch < 20:
            self.logger.experiment.add_image('Val_Sample', img_grid(images_noisy), self.current_epoch)

        image_logits, _ = self.forward(images_noisy)

        embed_clean = self.baseclip.encode_image(image_clean.type(torch.float32))
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = F.mse_loss(embed_clean, embed_noisy)

        top_1 = top_k_accuracy(image_logits, labels, k=1)
        top_5 = top_k_accuracy(image_logits, labels, k=5)

        loss_dict = {
            "Val_Loss": loss,
            "Top_1": top_1,
            "Top_5": top_5
        }

        output = {
            'Val_Results': loss_dict,
            'log': loss_dict,
            'progress_bar': loss_dict
        }

        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['Val_Results']['Val_Loss'] for x in outputs]).mean()
        top_1_mean = torch.stack([x['Val_Results']['Top_1'] for x in outputs]).sum() / self.N_val
        top_5_mean = torch.stack([x['Val_Results']['Top_5'] for x in outputs]).sum() / self.N_val

        loss_dict = {
            'Val_loss': val_loss_mean,
            'Top_1': top_1_mean,
            'Top_5': top_5_mean
        }

        output = {
            'Val_Loss': val_loss_mean,
            'log': loss_dict,
            'progress_bar': loss_dict
        }

        return output

def run_noisy_clip():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    seed_everything(args.seed)

    dataset = ImageNetCLIPDataset(args)
    dataset.setup()
    model = NoisyCLIP(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir= os.getcwd(),
        version=args.experiment_name,
        name='Logs'
    )
    trainer.logger = logger

    trainer.fit(model, dataset)


if __name__ == "__main__":
    run_noisy_clip()
