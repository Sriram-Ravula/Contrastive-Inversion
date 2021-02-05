#!/usr/bin/env python

import torch
from torch import Tensor
import typing
import torch.nn.functional as F

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

class ModifiedCLIP(torch.nn.Module):
  def __init__(self, baseclip, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
    super(ModifiedCLIP, self).__init__()
    self.baseclip =  baseclip.eval()
    self.noisy_visual_encoder = clip.model.VisualTransformer(
                                  input_resolution=input_resolution,
                                  patch_size=patch_size,
                                  width=width,
                                  layers=layers,
                                  heads=heads,
                                  output_dim=output_dim
                                )
    
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

  def fit(train_dataloader, valid_dataloader=None, epochs=10):
    loss_fun = ContrastiveLoss()
    optim = torch.optim.SGD(self.noisy_visual_encoder.parameters(), lr=1e-3, momentum=0.9)
    for t in epochs:
      for i, (image_clean, image_noisy) in train_dataloader:
        optim.zero_grad()
        embed_clean = baseclip.encode_image(image_clean)
        embed_noisy = self.encode_noisy_image(image_noisy)
        loss = loss_fun(embed_clean, embed_noisy)
        loss.backward()
        optimizer.step()

    return None