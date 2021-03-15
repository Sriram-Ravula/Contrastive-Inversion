#!/usr/bin/env python

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

import torchvision

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data  import random_split, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import CIFAR10

class ContrastiveLoss(torch.nn.modules.loss._WeightedLoss):
    r"""Contrastive loss within a batch. The loss calculated tries to maximize
    similarity between clean and noisy versions of the same image, while minimizing
    similarity of clean and noisy versions of different images.
    """
    def __init__(self, weight: typing.Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean', tau=1, device='cpu') -> None:
        super(ContrastiveLoss, self).__init__(weight, size_average, reduce, reduction)
        self.tau = tau
        self.device = device

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        bsz = input1.shape[0]
        full_tensor = torch.cat([input1.unsqueeze(1),input2.unsqueeze(1)], dim=1).view(2*bsz, 1, -1)
        tensor1 = full_tensor.expand(2*bsz,2*bsz,-1)
        tensor2 = full_tensor.permute(1,0,2).expand(2*bsz,2*bsz,-1)
        sim_mat = torch.nn.CosineSimilarity(dim=-1)(tensor1,tensor2)
        exp_sim_mat = torch.exp(sim_mat/self.tau)
        mask = torch.ones_like(exp_sim_mat) - torch.eye(2*bsz).to(self.device)
        logmat = -torch.log(exp_sim_mat)+torch.log(torch.sum(mask*exp_sim_mat, 1))

        loss = (torch.sum(torch.diag(logmat, diagonal=1)) + torch.sum(torch.diag(logmat, diagonal=-1)))/2
        return loss/bsz if self.reduction == 'mean' else loss

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, transform_clean=None, transform_noisy=None):
        self.base = clean_dataset
        self.transform_clean = transform_clean
        self.transform_noisy = transform_noisy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, _ = self.base[idx]

        image_clean = self.transform_clean(image_orig) if self.transform_clean is not None else image_orig
        image_noisy = self.transform_noisy(image_orig) if self.transform_noisy is not None else image_orig

        return image_clean, image_noisy

class CIFAR10CLIPDataset(LightningDataModule):
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
    def __init__(self, args):
        super(ImageNetCLIPDataset, self).__init__()

        self.hparams = args
        # self.noise_transform = self.hparams.noise_transform
        self.dataset_dir = self.hparams.dataset_dir
        self.batch_size = self.hparams.batch_size


    def setup(self, stage=None):
        train_data = torchvision.datasets.ImageNet(
        	root=self.hparams.dataset_dir,
            split="train",
            transform=None
        )
        val_data = torchvision.datasets.ImageNet(
            root=self.hparams.dataset_dir,
            split="val",
            transform=ImageNetSquareMaskVal()
        )

        filename = self.hparams.dataset_dir + self.hparams.subset_file_name

        train_data, og_to_new_dict, text_labels = get_subset(train_data, filename=filename, return_class_labels=True)
        self.val_data, _ = get_subset(val_data, filename=filename)

        self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, transform_clean=ImageNetBaseTransform, transform_noisy=ImageNetSquareMask)

        if self.hparams.save_mapping_and_text:
            pickle.dump((og_to_new_dict, text_labels), open(self.hparams.mapping_and_text_file, 'wb'))

    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2*self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False)


class NoisyCLIP(LightningModule):
    def __init__(self, args):
        r'''For now, this only creates a separate copy of the visual transformer,
        using the same architecture as the one for clean images.
        '''
        super(NoisyCLIP, self).__init__()
        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        if self.hparams.dataset == "Imagenet-100":
            self.N_val = 50000 # Default ImageNet validation set.
            if self.hparams.mapping_and_text_file is None:
                raise ValueError('No file from which to read mapping/text labels was specified.')

            og_to_new_dict, text_labels = pickle.load(open(self.hparams.mapping_and_text_file, 'rb'))
            self.class_map = og_to_new_dict
            text_list = ['A photo of '+label.strip().replace('_',' ') for label in text_labels]

        self.logit_scale = self.hparams.logit_scale
        self.baseclip = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0]
        self.baseclip.eval()
        self.text_embeddings = self.baseclip.encode_text(clip.tokenize(text_list).to(self.hparams.device))
        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0].visual
        self.noisy_visual_encoder.train()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=1e-4, momentum=0.7)
        return optim

    def contrastive_loss(self, input1, input2, tau=1, reduction='mean'):
        bsz = input1.shape[0]
        full_tensor = torch.cat([input1.unsqueeze(1),input2.unsqueeze(1)], dim=1).view(2*bsz, 1, -1)
        tensor1 = full_tensor.expand(2*bsz,2*bsz,-1)
        tensor2 = full_tensor.permute(1,0,2).expand(2*bsz,2*bsz,-1)
        sim_mat = torch.nn.CosineSimilarity(dim=-1)(tensor1,tensor2)
        exp_sim_mat = torch.exp(sim_mat/tau)
        mask = torch.ones_like(exp_sim_mat) - torch.eye(2*bsz).to(self.device)
        logmat = -torch.log(exp_sim_mat)+torch.log(torch.sum(mask*exp_sim_mat, 1))

        loss = (torch.sum(torch.diag(logmat, diagonal=1)) + torch.sum(torch.diag(logmat, diagonal=-1)))/2
        return loss/bsz if reduction == 'mean' else loss

    def encode_noisy_image(self, image):
        return self.noisy_visual_encoder(image.type(torch.float16))

    def forward(self, image, text=None):
        r'''Forward method taken from original CLIP model. The use is the same as
        the original function, apart from using the encoder trained for noise images.
        '''
        image_features = self.encode_noisy_image(image)
        text_features = self.text_embeddings

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features.type(torch.float16) @ text_features.type(torch.float16).t() # Funny thing, here the original code spells 'iamge' instead of image. Hidden copyright protection? :p
        logits_per_text = self.logit_scale * text_features.type(torch.float16) @ image_features.type(torch.float16).t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def training_step(self, train_batch, batch_idx):
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
        images_noisy, labels = test_batch
        image_logits, _ = self.forward(images_noisy)
        preds = torch.argmax(image_logits.softmax(dim=-1), axis=1)

        if self.hparams.dataset == "Imagenet-100":
            labels = map_classes(labels, self.class_map)

        if batch_idx == 0 and self.current_epoch < 20:
            self.logger.experiment.add_image('Val_Sample', img_grid(x), self.current_epoch)

        image_logits, _ = self.forward(images_noisy)

        # loss = self.criterion(logits, labels)
        top_1 = top_k_accuracy(logits, labels, k=1)
        top_5 = top_k_accuracy(logits, labels, k=5)

        loss_dict = {
            # "Val_Loss": loss,
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
        val_loss_mean = torch.stack([x['Val_Loss']['Val_Loss'] for x in outputs]).mean()
        top_1_mean = torch.stack([x['Val_Loss']['Top_1'] for x in outputs]).sum() / self.N_val
        top_5_mean = torch.stack([x['Val_Loss']['Top_5'] for x in outputs]).sum() / self.N_val

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

    config = yaml_config_hook("./config/config_noisy_clip.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    seed_everything(args.seed)

    dataset = ImageNetCLIPDataset(args)

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
    run_baseline()
