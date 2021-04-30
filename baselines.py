import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

import argparse
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy

from utils import *
import clip

class CLIP_finetune(nn.Module):
    def __init__(self, args):
        super(CLIP_finetune, self).__init__()

        self.args = args

        #grab the clip model
        self.feature_extractor = clip.load(args.clip_model, device='cpu', jit=False)[0].visual

         #grab the input dimension to the final layer
        if args.clip_model == 'ViT-B/32' or args.clip_model == 'RN101':
            num_filters = 512
        elif args.clip_model == 'RN50':
            num_filters = 1024
        elif args.clip_model == 'RN50x4':
            num_filters = 640
        else:
            raise ValueError("Unsupported CLIP model selected.")
        
        self.classifier = nn.Linear(num_filters, args.num_classes) 

        #If we are only training the classifier, then freeze the backbone!
        if args.freeze_backbone:
            self.feature_extractor.eval()

            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor.train()
        
    def forward(self, x):
        if self.args.freeze_backbone:
            self.feature_extractor.eval()

            with torch.no_grad():
                features = self.feature_extractor(x).flatten(1).float()
        
        else:
            self.feature_extractor.train()
            features = self.feature_extractor(x).flatten(1).float()
        
        x = self.classifier(features)

        return x

class RESNET_finetune(nn.Module):
    def __init__(self, args):
        super(RESNET_finetune, self).__init__()

        self.args = args

        #Grab the correct Resnet model
        if args.resnet_model == "50":
            backbone = models.resnet50(pretrained=True)
        elif args.resnet_model == "101":
            backbone = models.resnet101(pretrained=True)

        #grab the input dimension to the final layer
        num_filters = backbone.fc.in_features

        #define the feature extraction module
        layers = list(backbone.children())[:-1] #leave out the fc layer!
        self.feature_extractor = nn.Sequential(*layers)
        
        self.classifier = nn.Linear(num_filters, args.num_classes) 

        #If we are only training the classifier, then freeze the backbone!
        if args.freeze_backbone:
            self.feature_extractor.eval()

            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor.train()

    def forward(self, x):
        if self.args.freeze_backbone:
            self.feature_extractor.eval()

            with torch.no_grad():
                features = self.feature_extractor(x).flatten(1)
        
        else:
            self.feature_extractor.train()
            features = self.feature_extractor(x).flatten(1)
        
        x = self.classifier(features)

        return x


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
        else:
            if self.hparams.distortion == "None":
                self.train_set_transform = ImageNetBaseTransform(self.hparams)
                self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
            else:
                #If we are using the ImageNet dataset, then set up the train and val sets to use the same mask if needed! 
                self.train_set_transform = ImageNetDistortTrain(self.hparams)
            
                if self.hparams.fixed_mask:        
                    self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
                else:
                    self.val_set_transform = ImageNetDistortVal(self.hparams)

        #(2) Grab the correct baseline pre-trained model
        if self.hparams.encoder == 'resnet':
            self.encoder = RESNET_finetune(self.hparams)
        elif self.hparams.encoder == 'clip':
            self.encoder = CLIP_finetune(self.hparams)
        else:
            raise ValueError("Please select a valid encoder model.")

        #(3) Set up our criterion - here we use reduction as "sum" so that we are able to average over all validation sets
        self.criterion = nn.CrossEntropyLoss(reduction = "sum")

        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)

        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

        self.test_top_1 = Accuracy(top_k=1)
        self.test_top_5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.encoder(x)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.encoder.parameters(), lr = self.lr)

        if self.hparams.dataset == "ImageNet100":
            num_steps = 126689//(self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters
        else:
            num_steps = 500

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

        return [opt], [scheduler]

    #DATALOADERS
    def train_dataloader(self):
        if self.hparams.dataset == "ImageNet100":
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
            val_dataset = ImageNet100(
                root=self.hparams.dataset_dir,
                split = 'val',
                transform = self.val_set_transform
            )

            self.N_val = 5000

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

        test_dataloader = DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)

        return test_dataloader
    
    #TRAINING
    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Train_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, sync_dist_op='sum')
        self.log("train_top_1", self.train_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("train_top_5", self.train_top_5(pred_probs, y), prog_bar=False, logger=False)

        return loss
    
    def training_epoch_end(self, outputs):
        self.log("train_top_1", self.train_top_1.compute(), prog_bar=True, logger=True)
        self.log("train_top_5", self.train_top_5.compute(), prog_bar=True, logger=True)

    #VALIDATION
    def validation_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Val_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        self.log("val_top_1", self.val_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("val_top_5", self.val_top_5(pred_probs, y), prog_bar=False, logger=False)
    
    def validation_epoch_end(self, outputs):
        self.log("val_top_1", self.val_top_1.compute(), prog_bar=True, logger=True)
        self.log("val_top_5", self.val_top_5.compute(), prog_bar=True, logger=True)
    
    #TESTING
    def test_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.experiment.add_image('Test_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)
        pred_probs = logits.softmax(dim=-1)

        self.log("test_top_1", self.test_top_1(pred_probs, y), prog_bar=False, logger=False)
        self.log("test_top_5", self.test_top_5(pred_probs, y), prog_bar=False, logger=False)
    
    def test_epoch_end(self, outputs):
        self.log("test_top_1", self.test_top_1.compute(), prog_bar=True, logger=True)
        self.log("test_top_5", self.test_top_5.compute(), prog_bar=True, logger=True)

def run_baseline(config_file, lr = 0):
    args = grab_config(config_file)

    if lr == 0:
        lr = args.lr
    else:
        args.lr = lr
    
    seed_everything(args.seed)

    model = Baseline(args)

    logger = TensorBoardLogger(
        save_dir= args.logdir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger)      

    trainer.fit(model)
    #trainer.test(model)

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

if __name__ == "__main__":

    run_baseline("Clean.yaml")
