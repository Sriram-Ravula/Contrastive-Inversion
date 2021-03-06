import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
from utils import ImageNetSquareMask, ImageNetSquareMaskVal, img_grid, yaml_config_hook, top_k_accuracy, get_subset, map_classes, ImageNetRandomMask, ImageNetRandomMaskVal
import clip

class Baseline(LightningModule):
    def __init__(self, args):
        super(Baseline, self).__init__()

        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        if self.hparams.dataset == "Imagenet-100":
            self.class_map = None

        if self.hparams.encoder == 'resnet':
            #(1) load a resnet model with or without ImageNet Pre-Training
            self.encoder = models.resnet50(pretrained=self.hparams.pretrained)

            #(2) replace the last, linear layer of Resnet with one that has appropriate dimension for CIFAR-10
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.hparams.num_classes) #replace the resnet output with correct number of classes

        elif self.hparams.encoder == 'clip':
            clip_type = self.hparams.clip_model

            #Extract the visual encoder from the pre-trained CLIP
            self.baseclip = clip.load(clip_type, device='cpu', jit=False)[0].visual
            self.baseclip.train()

            if clip_type == 'ViT-B/32':
                self.output = nn.Linear(512, self.hparams.num_classes)
            elif clip_type == 'RN50':
                self.output = nn.Linear(1024, self.hparams.num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if self.hparams.encoder == 'clip':
            n = x.size(0)
            x = self.baseclip(x)

            return self.output(x.view(n, -1).float())

        elif self.hparams.encoder == 'resnet':
            return self.encoder(x)
    
    def configure_optimizers(self):
        if self.hparams.encoder == 'clip':
            #opt = torch.optim.Adam(list(self.baseclip.parameters()) + list(self.output.parameters()), lr = self.hparams.lr)
            opt = torch.optim.SGD(list(self.baseclip.parameters()) + list(self.output.parameters()), lr = self.hparams.lr, momentum=0.7)

        elif self.hparams.encoder == 'resnet':
            opt = torch.optim.Adam(self.encoder.parameters(), lr = self.hparams.lr)

        return opt

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.dataset == "Imagenet-100":
            y = map_classes(y, self.class_map)

        logits = self.forward(x)

        loss = self.criterion(logits, y)

        loss_dict = {"Train_Loss": loss}

        output = {
            'loss': loss,
            'progress_bar': loss_dict,
            'log': loss_dict
        }

        return output

    def train_dataloader(self):
        if self.hparams.dataset == "Imagenet-100":
            if self.hparams.distortion == "squaremask":
                transform = ImageNetSquareMask(mask_length = self.hparams.mask_length)
            elif self.hparams.distortion == "randommask":
                transform = ImageNetRandomMask(percent_missing = self.hparams.percent_missing)
            
            train_dataset = torchvision.datasets.ImageNet(
                root=self.hparams.dataset_dir,
                split="train",
                transform=transform
            )

            filename = self.hparams.dataset_dir + self.hparams.subset_file_name

            train_dataset, og_to_new_dict = get_subset(train_dataset, filename=filename)

            if self.class_map == None:
                self.class_map = og_to_new_dict
                #print("\n\nSET CLASS MAP\n\n") #TODO REMOVE THIS DEBUG

        train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        if self.hparams.dataset == "Imagenet-100":
            if self.hparams.distortion == "squaremask":
                transform = ImageNetSquareMaskVal(mask_length = self.hparams.mask_length)
            elif self.hparams.distortion == "randommask":
                transform = ImageNetRandomMaskVal(percent_missing = self.hparams.percent_missing)

            val_dataset = torchvision.datasets.ImageNet(
                root=self.hparams.dataset_dir,
                split="val",
                transform=transform
            )

            filename = self.hparams.dataset_dir + self.hparams.subset_file_name

            val_dataset, og_to_new_dict = get_subset(val_dataset, filename=filename)

            if self.class_map == None:
                self.class_map = og_to_new_dict
                #print("\n\nSET CLASS MAP\n\n") #TODO REMOVE THIS DEBUG

        self.N_val = len(val_dataset)

        val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)

        return val_dataloader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.hparams.dataset == "Imagenet-100":
            y = map_classes(y, self.class_map)

        if batch_idx == 0 and self.current_epoch < 15:
            self.logger.experiment.add_image('Val_Sample', img_grid(x), self.current_epoch)

        logits = self.forward(x)

        loss = self.criterion(logits, y)
        top_1 = top_k_accuracy(logits, y, k=1)
        top_5 = top_k_accuracy(logits, y, k=5)

        loss_dict = {
            "Val_Loss": loss,
            "Top_1": top_1,
            "Top_5": top_5
        }

        output = {
            'Val_Loss': loss_dict,
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

def run_baseline():
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config_baseline.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    seed_everything(args.seed)

    model = Baseline(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir= args.log_dir,
        version=args.experiment_name,
        name='Contrastive-Inversion'
    )
    trainer.logger = logger
    trainer.num_sanity_val_steps=-1 #Run an entire validation epoch before fine-tuning

    trainer.fit(model)


if __name__ == "__main__":
    run_baseline()
