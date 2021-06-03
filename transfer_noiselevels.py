#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch

from torchvision.datasets import CIFAR10, CIFAR100, STL10

from utils import *

from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data  import DataLoader

from transfer_learning import TransferLearning

class TransferTestDataset(LightningDataModule):
    """
    Dataset for transfer learning, to be used only for testing.
    Implemented separately because it is linked to both end-to-end suprevised and contrastive training.

    """
    def __init__(self, args):
        super(TransferTestDataset, self).__init__()

        self.hparams = args

        if self.hparams.dataset == "CIFAR10" or self.hparams.dataset == "CIFAR100" or self.hparams.dataset == "STL10":
            #Get the correct image transform
            if self.hparams.distortion == "None":
                self.val_set_transform = GeneralBaseTransformVal(self.hparams)
            elif self.hparams.distortion == "multi":
                self.val_set_transform = GeneralDistortValMulti(self.hparams)
            else:
                self.val_set_transform = GeneralDistortVal(self.hparams)

        elif self.hparams.dataset == 'COVID' or self.hparams.dataset == 'ImageNet100B' or self.hparams.dataset == 'imagenet-100B':
            #Get the correct image transform
            if self.hparams.distortion == "None":
                self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
            elif self.hparams.distortion == "multi":
                self.val_set_transform = ImageNetDistortValMulti(self.hparams)
            else:
                self.val_set_transform = ImageNetDistortVal(self.hparams)

    def _grab_dataset(self):
        transform = self.val_set_transform

        if self.hparams.dataset == "CIFAR10":
            dataset = CIFAR10(root=self.hparams.dataset_dir, train=False, transform=transform, download=True)

        elif self.hparams.dataset == "CIFAR100":
            dataset = CIFAR100(root=self.hparams.dataset_dir, train=False, transform=transform, download=True)

        elif self.hparams.dataset == 'STL10':
            dataset = STL10(root=self.hparams.dataset_dir, split='test', transform = transform, download=True)

        elif self.hparams.dataset == 'COVID':
            dataset = torchvision.datasets.ImageFolder(root = self.hparams.dataset_dir + 'test', transform=transform)

        elif self.hparams.dataset == 'ImageNet100B' or self.hparams.dataset == 'imagenet-100B':
            dataset = ImageNet100(root = self.hparams.dataset_dir, split='val', transform=transform)

        return dataset

    def setup(self, stage=None):
        self.val_data = self._grab_dataset()

    def test_dataloader(self):
        #SHUFFLE TRUE FOR COVID AUROC STABILITY
        return DataLoader(self.val_data, batch_size=512, num_workers=self.hparams.workers, worker_init_fn=(lambda wid: np.random.seed(int(torch.rand(1)[0]*1e6) + wid)), pin_memory=True, shuffle=True)

    def predict_dataloader(self):
        #SHUFFLE TRUE FOR COVID AUROC STABILITY
        return DataLoader(self.val_data, batch_size=512, num_workers=self.hparams.workers, worker_init_fn=(lambda wid: np.random.seed(int(torch.rand(1)[0]*1e6) + wid)), pin_memory=True, shuffle=True)

def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def transfer_eval():
    args = grab_config()
    args.gpus = [0]

    seed_everything(42)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger, progress_bar_refresh_rate=0)

    #create necessary directories for saving results
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    if not os.path.exists(os.path.join(args.results_dir, args.experiment_name)):
        os.mkdir(os.path.join(args.results_dir, args.experiment_name))

    all_results = []

    #perform several tests over the dataset and aggregate the results
    for test in range(args.num_tests):
        saved_model = TransferLearning.load_from_checkpoint(args.checkpoint_path)

        test_data = TransferTestDataset(args)
        results = trainer.test(model=saved_model, datamodule=test_data, verbose=False)
        all_results.extend(results)

        print("Done with " + str(test))

    top1_accs = [x['test_top_1'] for x in all_results]

    if args.dataset != 'COVID':
        top5_accs = [x['test_top_5'] for x in all_results]
    else:
        auroc = [x['test_auc'] for x in all_results]

    results_file = os.path.join(args.results_dir, args.experiment_name, args.dataset + args.distortion + '.out')

    with open(results_file, 'w+') as f:
        f.write('Top 1 mean\t{0:.4f}\n'.format(np.mean(top1_accs)))
        f.write('Top 1 std\t{0:.4f}\n'.format(np.std(top1_accs, ddof=1)))
        f.write('Top 1 stderr\t{0:.4f}\n'.format(np.std(top1_accs, ddof=1)/np.sqrt(args.num_tests)))
        if args.dataset != 'COVID':
            f.write('Top 5 mean\t{0:.4f}\n'.format(np.mean(top5_accs)))
            f.write('Top 5 std\t{0:.4f}\n'.format(np.std(top5_accs, ddof=1)))
            f.write('Top 5 stderr\t{0:.4f}\n'.format(np.std(top5_accs, ddof=1)/np.sqrt(args.num_tests)))
        else:
            f.write('AUROC 5 mean\t{0:.4f}\n'.format(np.mean(auroc)))
            f.write('AUROC 5 std\t{0:.4f}\n'.format(np.std(auroc, ddof=1)))
            f.write('AUROC 5 stderr\t{0:.4f}\n'.format(np.std(auroc, ddof=1)/np.sqrt(args.num_tests)))

if __name__ == "__main__":
    transfer_eval()
