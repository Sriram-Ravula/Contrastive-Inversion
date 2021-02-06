#!/usr/bin/env python

import torch
from torch import Tensor
import typing
import torch.nn.functional as F
import model
import copy
from tqdm import tqdm

class ContrastiveLoss(torch.nn.modules.loss._WeightedLoss):
    r"""Contrastive loss within a batch. The loss calculated tries to maximize
    similarity between clean and noisy versions of the same image, while minimizing
    similarity of clean and noisy versions of different images.
    """
    def __init__(self, weight: typing.Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean', tau=1) -> None:
        super(ContrastiveLoss, self).__init__(weight, size_average, reduce, reduction)
        self.tau = tau

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        bsz = input1.shape[0]
        tensor1 = input1.unsqueeze(0).expand(bsz,bsz,-1)
        tensor2 = input2.unsqueeze(1).expand(bsz,bsz,-1)
        sim_mat = torch.nn.CosineSimilarity(dim=-1)(tensor1,tensor2)
        exp_sim_mat = torch.exp(sim_mat/self.tau)
        print(exp_sim_mat)
        mask = torch.ones_like(exp_sim_mat) - torch.eye(bsz)
        logmat = -torch.log(exp_sim_mat)+torch.log(torch.sum(mask*exp_sim_mat, 1))
        loss = torch.sum(torch.diag(logmat))+bsz
        return loss/bsz if self.reduction == 'mean' else loss

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, transform_clean=None, transform_noisy=None):
        self.base = clean_dataset
        self.transform_clean = transform_clean
        self.transform_noisy = transform_noisy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        if self.transform_clean:
            image_clean = self.transform_clean(image)
        if self.transform_noisy:
            image_noisy = self.transform_noisy(image)
        return image_clean, image_noisy


class ModifiedCLIP(torch.nn.Module):
    def __init__(self, baseclip, device='cpu'):
        r'''For now, this only creates a separate copy of the visual transformer,
        using the same architecture as the one for clean images.
        '''
        super(ModifiedCLIP, self).__init__()
        self.baseclip =  baseclip.eval().to(device)
        self.dtype = baseclip.visual.conv1.weight.dtype

        # Somehow derive parameters from base clip
        self.noisy_visual_encoder = model.VisualTransformer(
                                            baseclip.input_resolution,
                                            3,
                                            baseclip.visual.class_embedding.shape[0],
                                            3,
                                            3,
                                            baseclip.visual.proj.shape[1]
                                        )
        self.noisy_visual_encoder.train()

    def encode_noisy_image(self, image):
        return self.noisy_visual_encoder(image.type(self.dtype))

    def forward(self, image, text):
        r'''Forward method taken from original CLIP model. The use is the same as
        the original function, apart from using the encoder trained for noise images.
        '''
        image_features = self.encode_noisy_image(image)
        text_features = baseclip.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = baseclip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() # Funny thing, here the original code spells 'iamge' instead of image. Hidden copyright protection? :p
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def fit(self, train_dataloader, valid_dataloader=None, epochs=10):
        loss_fun = ContrastiveLoss()
        optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=1e-3, momentum=0.9)
        for t in range(epochs):
            self.noisy_visual_encoder.train()
            for image_clean, image_noisy in tqdm(train_dataloader):
                optim.zero_grad()
                embed_clean = self.baseclip.encode_image(image_clean)
                embed_noisy = self.encode_noisy_image(image_noisy)
                loss = loss_fun(embed_clean, embed_noisy)
                loss.backward()
                optimizer.step()

            if valid_dataloader:
                self.noisy_visual_encoder.eval()
                with torch.no_grad():
                    for i, (image_clean, image_noisy) in valid_dataloader:
                        embed_clean = baseclip.encode_image(image_clean)
                        embed_noisy = self.encode_noisy_image(image_noisy)
                        loss = loss_fun(embed_clean, embed_noisy)

        return None

    def predict(self, test_dataloader, text_labels):
        text = clip.tokenize(['a picture of a '+label for label in text_labels]).to(device)
        integer_labels = {}
        for i,label in enumerate(text_labels):
            integer_labels[label] = i
        self.noisy_visual_encoder.eval()
        total = 0
        corr = 0
        with torch.no_grad():
            for _, (images_noisy, labels) in test_dataloader:
                image_logits, _ = self.forward(images_noisy, text)
                preds = image_logits.softmax(dim=-1)
                total += len(preds)
                corr += np.sum(preds==np.array([integer_labels[label] for label in labels]))

        return corr/total
