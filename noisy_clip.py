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

import pytorch_lightning as pl
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
    def __init__(self, clean_dataset, transform_noisy=None):
        self.base = clean_dataset
        self.transform_noisy = transform_noisy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_clean, _ = self.base[idx]
        if self.transform_noisy:
            image_noisy = self.transform_noisy(image_clean)
        return image_clean, image_noisy

class CIFAR10CLIPDataset(pl.LightningDataModule):
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



class ModifiedCLIP(pl.LightningModule):
    def __init__(self, baseclip_type, text_list, device='cpu'):
        r'''For now, this only creates a separate copy of the visual transformer,
        using the same architecture as the one for clean images.
        '''
        super(ModifiedCLIP, self).__init__()
        self.baseclip = clip.load(baseclip_type, device, jit=False)[0]
        self.baseclip.eval()
        self.text_embeddings = self.baseclip.encode_text(clip.tokenize(text_list).to(device))
        self.noisy_visual_encoder = clip.load(baseclip_type, device, jit=False)[0].visual
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
        logit_scale = 0.07
        logits_per_image = logit_scale * image_features.type(torch.float16) @ text_features.type(torch.float16).t() # Funny thing, here the original code spells 'iamge' instead of image. Hidden copyright protection? :p
        logits_per_text = logit_scale * text_features.type(torch.float16) @ image_features.type(torch.float16).t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def training_step(self, train_batch, batch_idx):
        image_clean, image_noisy = train_batch
        embed_clean = self.baseclip.encode_image(image_clean.type(torch.float32))
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = self.contrastive_loss(embed_clean, embed_noisy)
        # result = pl.TrainResult(loss)
        if batch_idx % 1 == 0:
            self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        image_clean, image_noisy = val_batch
        embed_clean = self.baseclip.encode_image(image_clean)
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = self.contrastive_loss(embed_clean, embed_noisy)
        # result = pl.EvalResult(loss)
        if batch_idx % 1 == 0:
            self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        images_noisy, labels = test_batch
        image_logits, _ = self.forward(images_noisy.to(self.device))
        preds = torch.argmax(image_logits.softmax(dim=-1), axis=1)
        acc = FM.accuracy(preds, labels)
        self.log('accuracy', acc)
        return acc

    # def fit(self, train_dataloader, valid_dataloader=None, epochs=5):
    #     loss_fun = ContrastiveLoss(device=self.device)
    #     optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=1e-4, momentum=0.7)
    #     for t in range(epochs):
    #         self.noisy_visual_encoder.train()
    #         for i, (image_clean, image_noisy) in enumerate(train_dataloader):
    #             optim.zero_grad()
    #             embed_clean = self.baseclip.encode_image(image_clean)
    #             embed_noisy = self.encode_noisy_image(image_noisy.to(self.device))
    #             loss = loss_fun(embed_clean.to(self.device), embed_noisy)
    #             if(i % 20 == 0):
    #                 print('Epoch {0:}, \tBatch {1:}\tLoss: {2:.5f}'.format(t+1, i+1, loss.item()))
    #             loss.backward()
    #             optim.step()

    #         if valid_dataloader:
    #             self.noisy_visual_encoder.eval()
    #             with torch.no_grad():
    #                 for embed_clean, image_noisy in valid_dataloader:
    #                     embed_noisy = self.encode_noisy_image(image_noisy)
    #                     loss = loss_fun(embed_clean.to(self.device), embed_noisy)

    #     return None

    # def score(self, test_dataloader, batches_per_epoch=100):
        
    #     self.noisy_visual_encoder.eval()
    #     total = 0
    #     corr = 0
    #     with torch.no_grad():
    #         for i, (images_noisy, labels) in tqdm(enumerate(test_dataloader)):
    #             if i >= batches_per_epoch:
    #                 break
    #             image_logits, _ = self.forward(images_noisy.to(self.device))
    #             preds = np.argmax(image_logits.softmax(dim=-1).cpu().numpy(), axis=1)
    #             total += len(preds)
    #             corr += np.sum(preds==np.array([label.item() for label in labels]))

    #     return corr/total
