#!/usr/bin/env python

import argparse
import numpy as np
import torch
import typing
import torch.nn.functional as F
from clip_files import model, clip
import pickle

from utils import *

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data  import DataLoader

from u2nets import BGMask
import spur_datasets.waterbirds as wb

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    """
    This class takes a dataset and creates a contrastive version of that dataset.
    Each item of the dataset is a tuple of a clean image and a noisy image (two
    separate transformations.)
    """
    def __init__(self, clean_dataset, transform_contrastive=None, return_label=False):
        self.base = clean_dataset
        self.transform_contrastive = transform_contrastive
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]
        image_clean, image_noisy = self.transform_contrastive(image_orig) if self.transform_contrastive is not None else (image_orig, image_orig)
        if self.return_label:
            return image_clean, image_noisy, label
        else:
            return image_clean, image_noisy

class ImageNetCLIPDataset(LightningDataModule):
    """
    Wrapper class for the ImageNet dataset, handles all data manipulations
    required in order to train the NoisyCLIP model.
    """
    def __init__(self, args):
        super(ImageNetCLIPDataset, self).__init__()

        self.hparams = args

        self.dataset_dir = self.hparams.dataset_dir
        self.batch_size = self.hparams.batch_size

        if self.hparams.distortion == "None":
            self.train_set_transform = ImageNetBaseTrainContrastive(self.hparams)
            self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
        elif self.hparams.distortion == 'multi':
            #set up the training transform and if we want a fixed mask, transfer the same mask to the validation transform
            self.train_set_transform = ImageNetDistortTrainMultiContrastive(self.hparams)
            self.val_set_transform = ImageNetDistortValMulti(self.hparams)
        else:
            #set up the training transform and if we want a fixed mask, transfer the same mask to the validation transform
            self.train_set_transform = ImageNetDistortTrainContrastive(self.hparams)

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

        self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, transform_contrastive=self.train_set_transform, return_label=True)

        # Save labels to be reused.
        if self.hparams.save_mapping_and_text:
            pickle.dump(text_labels, open(self.hparams.mapping_and_text_file, 'wb'))

    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2*self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False) # Only used for evaluation.

    def test_dataloader(self):
        return self.val_dataloader() # Same data to be used for testing, for our purposes.


class NoisyCLIPBirds(LightningModule):
    def __init__(self, args):
        """
        A class that trains OpenAI CLIP in a student-teacher fashion to classify distorted images.

        Given two identical pre-trained networks, Teacher - T() and Student S(), we freeze T() and train S().

        Given a batch of {x1, x2, ..., xN}, apply a given distortion to each one to obtain noisy images {y1, y2, ..., yN}.

        Feed original images to T() and obtain embeddings {T(x1), ..., T(xN)} and feed distorted images to S() and obtain embeddings {S(y1), ..., S(yN)}.

        Maximize the similarity between the pairs {(T(x1), S(y1)), ..., (T(xN), S(yN))} while minimizing the similarity between all non-matched pairs.
        """
        super(NoisyCLIPBirds, self).__init__()
        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        #(2) set up the teacher CLIP network - freeze it and don't use gradients!
        self.logit_scale = self.hparams.logit_scale
        self.baseclip = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0]
        self.baseclip.eval()
        self.baseclip.requires_grad_(False)
        self.clear_bg = BGMask()
        self.clear_bg.eval()
        self.clear_bg.requires_grad_(False)

        #(3) set up the student CLIP network - unfreeze it and use gradients!
        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0].visual
        self.noisy_visual_encoder.train()

        #(4) set up the training and validation accuracy metrics.
        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)
        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

    def criterion(self, input1, input2, reduction='mean'):
        """
        Args:
            input1: Embeddings of the clean/noisy images from the teacher/student. Size [N, embedding_dim].
            input2: Embeddings of the clean/noisy images from the teacher/student (the ones not used as input1). Size [N, embedding_dim].
            reduction: how to scale the final loss
        """
        bsz = input1.shape[0]

        # Use the simclr style InfoNCE
        if self.hparams.loss_type == 'simclr':
            # Create similarity matrix between embeddings.
            full_tensor = torch.cat([input1.unsqueeze(1),input2.unsqueeze(1)], dim=1).view(2*bsz, 1, -1)
            tensor1 = full_tensor.expand(2*bsz,2*bsz,-1)
            tensor2 = full_tensor.permute(1,0,2).expand(2*bsz,2*bsz,-1)
            sim_mat = torch.nn.CosineSimilarity(dim=-1)(tensor1,tensor2)

            # Calculate logits used for the contrastive loss.
            exp_sim_mat = torch.exp(sim_mat/self.hparams.loss_tau)
            mask = torch.ones_like(exp_sim_mat) - torch.eye(2*bsz).type_as(exp_sim_mat)
            logmat = -torch.log(exp_sim_mat)+torch.log(torch.sum(mask*exp_sim_mat, 1))

            #Grab the two off-diagonal similarities
            part1 = torch.sum(torch.diag(logmat, diagonal=1)[np.arange(0,2*bsz,2)])
            part2 = torch.sum(torch.diag(logmat, diagonal=-1)[np.arange(0,2*bsz,2)])

            #Take the mean of the two off-diagonals
            loss = (part1 + part2)/2

        #Use the CLIP-style InfoNCE
        elif self.hparams.loss_type == 'clip':
            # Create similarity matrix between embeddings.
            tensor1 = input1 / input1.norm(dim=-1, keepdim=True)
            tensor2 = input2 / input2.norm(dim=-1, keepdim=True)
            sim_mat = (1/self.hparams.loss_tau)*tensor1 @ tensor2.t()

            #Calculate the cross entropy between the similarities of the positive pairs, counted two ways
            part1 = F.cross_entropy(sim_mat, torch.LongTensor(np.arange(bsz)).to(self.device))
            part2 = F.cross_entropy(sim_mat.t(), torch.LongTensor(np.arange(bsz)).to(self.device))

            #Take the mean of the two off-diagonals
            loss = (part1+part2)/2

        #Take the simple MSE between the clean and noisy embeddings
        elif self.hparams.loss_type == 'mse':
            return F.mse_loss(input2, input1)

        else:
            raise ValueError('Loss function not understood.')

        return loss/bsz if reduction == 'mean' else loss


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.noisy_visual_encoder.parameters(), lr=self.hparams.lr)
        num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)
        return [optim], [sched]

    def encode_noisy_image(self, image):
        """
        Return S(yi) where S() is the student network and yi is distorted images.
        """
        return self.noisy_visual_encoder(image.type(torch.float16))

    def forward(self, image_features, text=None):
        """
        Given a set of noisy image embeddings, calculate the cosine similarity (scaled by temperature) of each image with each class text prompt.
        Calculates the similarity in two ways: logits per image (size = [N, n_classes]), logits per text (size = [n_classes, N]).
        This is mainly used for validation and classification.

        Args:
            image_features: the noisy image embeddings S(yi) where S() is the student and yi = Distort(xi). Shape [N, embedding_dim]
        """

        #load the pre-computed text features and load them on the correct device
        text_features = self.baseclip.encode_text(clip.tokenize(self.text_list).to(self.device))

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features.type(torch.float16) @ text_features.type(torch.float16).t()
        logits_per_text = self.logit_scale * text_features.type(torch.float16) @ image_features.type(torch.float16).t()

        return logits_per_image, logits_per_text

    # Training methods - here we are concerned with contrastive loss (or MSE) between clean and noisy image embeddings.
    def training_step(self, train_batch, batch_idx):
        """
        Takes a batch of clean and noisy images and returns their respective embeddings.

        Returns:
            embed_clean: T(xi) where T() is the teacher and xi are clean images. Shape [N, embed_dim]
            embed_noisy: S(yi) where S() is the student and yi are noisy images. Shape [N, embed_dim]
        """
        #clear_bg = BGMask().to(self.device).eval()
        image_orig, labels = train_batch
        image_no_bg = self.clear_bg(image_orig)
        embed_clean = self.baseclip.encode_image(image_no_bg)
        embed_noisy = self.encode_noisy_image(image_orig)
        return {'embed_clean': embed_clean, 'embed_noisy': embed_noisy}

    def training_step_end(self, outputs):
        """
        Given all the clean and noisy image embeddings form across GPUs from training_step, gather them onto a single GPU and calculate overall loss.
        """
        embed_clean_full = outputs['embed_clean']
        embed_noisy_full = outputs['embed_noisy']
        loss = self.criterion(embed_clean_full, embed_noisy_full)
        self.log('train_loss', loss, prog_bar=False, logger=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss

def run_noisy_clip():
    args = grab_config()

    seed_everything(args.seed)

    dataset = wb.BirdModule(root_dir=args.dataset_dir, groups=[0,1,2,3], batch_size=args.batch_size)
    dataset.setup()
    model = NoisyCLIPBirds(args)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    if not hasattr(args, 'increasing') or not args.increasing:
        trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[ModelCheckpoint(save_top_k=-1, period=25)])
        trainer.fit(model, datamodule=dataset)
    else:
        trainer = Trainer.from_argparse_args(args, logger=logger)
        trainer.fit(model)

def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    run_noisy_clip()