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
from utils import ImageNetSquareMask, ImageNetSquareMaskVal, img_grid, yaml_config_hook, top_k_accuracy, get_subset, map_classes

class Baseline(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        if self.hparams.dataset == "Imagenet-100":
            self.class_map = None

        if self.hparams.encoder == 'resnet':
            #(1) load a resnet model with or without ImageNet Pre-Training
            self.encoder = models.resnet50(pretrained=self.hparams.pretrained)

            """
            #(2) freeze or unfreeze parameters depending on if we would like to fine-tune the model or just do linear probe
            for param in self.encoder.parameters():
                param.requires_grad=self.hparams.finetune
            """

            #(3) replace the last, linear layer of Resnet with one that has appropriate dimension for CIFAR-10
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.hparams.num_classes) #replace the resnet output with correct number of classes
        
        """
        #Figure out how to get CLIP finetuned!
        elif self.hparams.encoder == 'clip-resnet':
            model, preprocess = clip.load('RN50', device='cpu', jit=False)
        """

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.encoder(x)
    
    def configure_optimizers(self):
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
            train_dataset = torchvision.datasets.ImageNet(
                root=self.hparams.dataset_dir,
                split="train",
                transform=ImageNetSquareMask()
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
            val_dataset = torchvision.datasets.ImageNet(
                root=self.hparams.dataset_dir,
                split="val",
                transform=ImageNetSquareMaskVal()
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

        if batch_idx == 0 and self.current_epoch < 20:
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
        save_dir= os.getcwd(),
        version=args.experiment_name,
        name='Logs'
    )
    trainer.logger = logger

    trainer.fit(model)


if __name__ == "__main__":
    run_baseline()
