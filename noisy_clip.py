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
    def __init__(self, clean_dataset, clean_embeddings, transform_noisy=None):
        self.base = clean_dataset
        self.clean_embeddings = clean_embeddings
        self.transform_noisy = transform_noisy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        if self.transform_noisy:
            image_noisy = self.transform_noisy(image)
        return self.clean_embeddings[idx], image_noisy


class ModifiedCLIP(torch.nn.Module):
    def __init__(self, baseclip_type, text_embeddings, device='cpu'):
        r'''For now, this only creates a separate copy of the visual transformer,
        using the same architecture as the one for clean images.
        '''
        super(ModifiedCLIP, self).__init__()
        self.text_embeddings = text_embeddings
        self.device = device

        # Somehow derive parameters from base clip
        # self.noisy_visual_encoder = model.VisualTransformer(
        #                                     224,
        #                                     3,
        #                                     256,
        #                                     2,
        #                                     1,
        #                                     512
        #                                 ).to(device)
        self.noisy_visual_encoder = clip.load(baseclip_type, device)[0].visual
        self.noisy_visual_encoder.train()

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

    def fit(self, train_dataloader, valid_dataloader=None, epochs=10):
        loss_fun = ContrastiveLoss(device=self.device)
        optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=1e-4, momentum=0.7)
        for t in range(epochs):
            self.noisy_visual_encoder.train()
            for i, (embed_clean, image_noisy) in enumerate(train_dataloader):
                optim.zero_grad()
                embed_noisy = self.encode_noisy_image(image_noisy.to(self.device))
                loss = loss_fun(embed_clean.to(self.device), embed_noisy)
                if((i+1) % 100 == 0):
                    print('Epoch {0:}, \tBatch {1:}\tLoss: {2:.5f}'.format(t+1, i+1, loss.item()))
                loss.backward()
                optim.step()

            if valid_dataloader:
                self.noisy_visual_encoder.eval()
                with torch.no_grad():
                    for embed_clean, image_noisy in valid_dataloader:
                        embed_noisy = self.encode_noisy_image(image_noisy)
                        loss = loss_fun(embed_clean.to(self.device), embed_noisy)

        return None

    def score(self, test_dataloader):
        
        self.noisy_visual_encoder.eval()
        total = 0
        corr = 0
        with torch.no_grad():
            for i, (images_noisy, labels) in tqdm(enumerate(test_dataloader)):
                image_logits, _ = self.forward(images_noisy.to(self.device))
                preds = np.argmax(image_logits.softmax(dim=-1).cpu().numpy(), axis=1)
                total += len(preds)
                corr += np.sum(preds==np.array([label.item() for label in labels]))

        return corr/total
