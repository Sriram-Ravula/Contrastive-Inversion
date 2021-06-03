#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch

from utils import *

from pytorch_lightning import Trainer, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data  import DataLoader

from linear_probe import LinearProbe
from baselines import Baseline
from zeroshot_validation import NoisyCLIPTesting
from transfer_learning import TransferLearning


DISTORTIONS = ['blur', 'digital', 'extra', 'noise', 'weather']
SUB_DISTORTIONS = {'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
                   'digital': ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
                   'extra': ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise'],
                   'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
                    'weather': ['brightness', 'fog', 'frost', 'snow']
                    }
LEVELS = ['1','2','3','4','5']

class CIFARC_DATASET(LightningDataModule):
    def __init__(self, args, sub_distortion, level):
        super(CIFARC_DATASET, self).__init__()

        self.hparams = args

        self.sub_distortion = sub_distortion
        self.level = level

        self.dataset_dir = self.hparams.dataset_dir

        if self.hparams.dataset == "CIFAR10-C" or self.hparams.dataset == "CIFAR100-C":
            self.val_set_transform = GeneralBaseTransformVal(self.hparams)
        else:
            raise ValueError("Please select a valid Dataset")

    def _grab_dataset(self):
        dataset = CIFARC(self.dataset_dir, self.sub_distortion, self.level, self.val_set_transform)

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

def noise_level_eval():
    args = grab_config()
    args.gpus = [0] # Force evaluation in a single gpu.

    seed_everything(42)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger)

    if not os.path.exists(os.path.join(args.results_dir, args.experiment_name)):
        os.mkdir(os.path.join(args.results_dir, args.experiment_name))

    for distortion in DISTORTIONS:
        print(distortion)
        for sub_distortion in SUB_DISTORTIONS[distortion]:
            print(sub_distortion)
            top_1_list = []
            top_5_list = []

            if not os.path.exists(os.path.join(args.results_dir, args.experiment_name, distortion)):
                os.makedirs(os.path.join(args.results_dir, args.experiment_name, distortion))

            for level in LEVELS:
                print(level)

                saved_model = TransferLearning.load_from_checkpoint(args.checkpoint_path)

                test_data = CIFARC_DATASET(args, sub_distortion=sub_distortion, level=level)
                results = trainer.test(model=saved_model, datamodule=test_data, verbose=False)

                top1_accs = results[0]['test_top_1']
                top5_accs = results[0]['test_top_5'] #[x['test_top_5'] for x in results]

                print(top1_accs)

                top_1_list.extend([top1_accs])
                top_5_list.extend([top5_accs])

                with open(os.path.join(args.results_dir, args.experiment_name, distortion, sub_distortion + '.out'), 'a+') as f:
                    f.write(level + ':\t{0:.4f}'.format(top1_accs) + '\t{0:.4f}\n'.format(top5_accs))

            with open(os.path.join(args.results_dir, args.experiment_name, distortion, sub_distortion + '.out'), 'a+') as f:
                f.write('MEAN:\t{0:.4f}\t{1:.4f}\n'.format(np.mean(top_1_list), np.mean(top_5_list)))

if __name__ == "__main__":
    noise_level_eval()
