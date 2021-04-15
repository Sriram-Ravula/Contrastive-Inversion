import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models
from utils import img_grid, yaml_config_hook, top_k_accuracy, ImageNetDistortTrain, ImageNetDistortVal, ImageNet100, ImageNetBaseTransform, ImageNetBaseTransformVal
import clip
import numpy as np
import shutil

class CLIP_finetune(nn.Module):
    def __init__(self, args):
        super(CLIP_finetune, self).__init__()

        self.baseclip = clip.load(args.clip_model, device='cpu', jit=False)[0].visual
        self.baseclip.train()

        if args.clip_model == 'ViT-B/32' or args.clip_model == 'RN101':
            self.output = nn.Linear(512, args.num_classes)
        elif args.clip_model == 'RN50':
            self.output = nn.Linear(1024, args.num_classes)
        elif args.clip_model == 'RN50x4':
            self.output = nn.Linear(640, args.num_classes)
        else:
            raise ValueError("Unsupported CLIP model selected.")
        
    def forward(self, x):
        n = x.size(0)
        x = self.baseclip(x)

        return self.output(x.view(n, -1).float())

class RESNET_finetune(nn.Module):
    def __init__(self, args):
        super(RESNET_finetune, self).__init__()

        if args.resnet_model == "50":
            self.encoder = models.resnet50(pretrained=args.pretrained)
        elif args.resnet_model == "101":
            self.encoder = models.resnet101(pretrained=args.pretrained)

        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, args.num_classes) 

    def forward(self, x):
        return self.encoder(x)


class Baseline(LightningModule):
    def __init__(self, args):
        super(Baseline, self).__init__()

        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        self.lr = self.hparams.lr #initalise this specially as a tuneable parameter

        #(1) Set up the dataset
        #Here, we use a 100-class subset of ImageNet
        if self.hparams.dataset != "ImageNet100":
            raise ValueError("Unsupported dataset selected.")
        # else:
        #     #If we are using the ImageNet dataset, then set up the train and val sets to use the same mask if needed! 
        #     self.train_set_transform = ImageNetDistortTrain(self.hparams)
        
        #     if self.hparams.fixed_mask:        
        #         self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
        #     else:
        #         self.val_set_transform = ImageNetDistortVal(self.hparams)

        #(2) Grab the correct baseline pre-trained model
        if self.hparams.encoder == 'resnet':
            self.encoder = RESNET_finetune(self.hparams)
        elif self.hparams.encoder == 'clip':
            self.encoder = CLIP_finetune(self.hparams)
        else:
            raise ValueError("Please select a valid encoder model.")

        #(3) Set up our criterion - here we use reduction as "sum" so that we are able to average over all validation sets
        self.criterion = nn.CrossEntropyLoss(reduction = "sum")

    def forward(self, x):
        return self.encoder(x)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.encoder.parameters(), lr = self.lr)

        if self.hparams.dataset == "ImageNet100":
            num_steps = 126689//self.hparams.batch_size
        else:
            num_steps = 100

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    def train_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            if self.hparams.distortion == "None":
                train_dataset = ImageNet100(
                    root=self.hparams.dataset_dir,
                    split = 'train',
                    transform = ImageNetBaseTransform(self.hparams)
                )
            else:
                train_dataset = ImageNet100(
                    root=self.hparams.dataset_dir,
                    split = 'train',
                    transform = self.train_set_transform
                )

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            if self.hparams.distortion == "None":
                val_dataset = ImageNet100(
                    root=self.hparams.dataset_dir,
                    split = 'val',
                    transform = ImageNetBaseTransformVal(self.hparams)
                )
            else:
                val_dataset = ImageNet100(
                    root=self.hparams.dataset_dir,
                    split = 'val',
                    transform = self.val_set_transform
                )

        self.N_val = len(val_dataset)

        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)

        return val_dataloader

    def test_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
            test_dataset = ImageNet100(
                root=self.hparams.dataset_dir,
                split = 'val',
                transform = self.val_set_transform
            )

        self.N_test = len(test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)

        return test_dataloader
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Train_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)

        loss = self.criterion(logits, y)
        top_1 = top_k_accuracy(logits, y, k=1)
        top_5 = top_k_accuracy(logits, y, k=5)

        loss_dict = {
            "val_ce_loss": loss,
            "top_1": top_1,
            "top_5": top_5
        }

        return loss_dict
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_ce_loss'] for x in outputs]).sum() / self.N_val
        top_1_mean = torch.stack([x['top_1'] for x in outputs]).sum() / self.N_val
        top_5_mean = torch.stack([x['top_5'] for x in outputs]).sum() / self.N_val

        self.log("val_loss", 1 - top_5_mean, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True) #VAL_LOSS IS ACTUALLY (1 - TOP_5)
        self.log("top_1", top_1_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("top_5", top_5_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_ce_loss", val_loss_mean, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
            
        if batch_idx == 0:
            self.logger.experiment.add_image('Test_Sample', img_grid(x), self.current_epoch)
            
        logits = self.forward(x)

        loss = self.criterion(logits, y)
        top_1 = top_k_accuracy(logits, y, k=1)
        top_5 = top_k_accuracy(logits, y, k=5)

        loss_dict = {
            "test_ce_loss": loss,
            "test_top_1": top_1,
            "test_top_5": top_5
        }

        return loss_dict
    
    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_ce_loss'] for x in outputs]).sum() / self.N_test
        top_1_mean = torch.stack([x['test_top_1'] for x in outputs]).sum() / self.N_test
        top_5_mean = torch.stack([x['test_top_5'] for x in outputs]).sum() / self.N_test

        self.log("test_ce_loss", test_loss_mean, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log("test_top_1", top_1_mean, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log("test_top_5", top_5_mean, prog_bar=False, on_step=False, on_epoch=True, logger=True)

def run_baseline(config_file, lr):
    args = grab_config(config_file)

    args.lr = lr

    model = Baseline(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir= args.logdir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer.logger = logger

    trainer.fit(model)

def linesearch(config_file, lr):
    args = grab_config(config_file)

    #set these to be 1 for tuning
    args.max_epochs = 1
    args.check_val_every_n_epoch = 1
    args.lr = lr

    model = Baseline(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir= args.logdir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer.logger = logger

    trainer.fit(model)
    
    top_5 = trainer.logged_metrics["top_5"]

    return top_5

def grab_config(config_file):
    """
    Given a filename, grab the corresponsing config file and return it 
    """
    #Grab the argments
    parser = argparse.ArgumentParser(description="Contrastive-Inversion")

    config = yaml_config_hook("./config/Supervised_CLIP_Baselines/" + config_file)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def main():
    seed_everything(1234)

    lrs_tune = [1e-2, 1e-3, 1e-4, 1e-5]

    configurations = [
        "RN50_blur21.yaml"
        #"RN50_blur37.yaml",
        #"RN50_noise01.yaml",
        #"RN50_noise03.yaml",
        #"RN50_noise05.yaml"
        #"RN50_rand50.yaml",
        #"RN50_rand75.yaml",
        #"RN50_rand90.yaml",
        #"RN50_sq50.yaml",
        #"RN50_sq100.yaml"
    ]

    for config_file in configurations:
        #track the best top_5 loss and corresponsing learning rate
        print(config_file)

        top_5_best = 0
        lr_best = 1

        for lr in lrs_tune:
            print("TESTING LR: ", lr)
            top_5 = linesearch(config_file, lr)
            print("TOP 5 ACC: ", top_5)

            if top_5 > top_5_best:
                print("NEW BEST ACC")
                top_5_best = top_5
                lr_best = lr
        
        print("BEST LR: ", lr_best)

        run_baseline(config_file, lr)

if __name__ == "__main__":

    run_baseline("RN50_test.yaml", 1e-4)
