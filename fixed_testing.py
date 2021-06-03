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
from denoising_baselines import BaselineDenoiseDistortVal

class ImageNet100Test(LightningDataModule):
    """
    This class loads the validation set of ImageNet100, to be used only for testing.
    Implemented separately because it is linked to both end-to-end suprevised and contrastive training.
    """
    def __init__(self, args):
        super(ImageNet100Test, self).__init__()

        self.hparams = args

        self.dataset_dir = self.hparams.dataset_dir

        if self.hparams.distortion == "None":
            self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
        elif self.hparams.denoised:
            self.val_set_transform = BaselineDenoiseDistortVal(self.hparams)
        else:
            self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):
        self.val_data = ImageNet100(
            root=self.hparams.dataset_dir,
            split="val",
            transform=self.val_set_transform
        )

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=10, num_workers=self.hparams.workers, worker_init_fn=(lambda wid: np.random.seed(int(torch.rand(1)[0]*1e6) + wid)), pin_memory=True, shuffle=False)


def grab_config():
    """
    Given a filename, grab the corresponsing config file and return it 
    """
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        if not k == 'seed' and not k == 'distributed_backend':
            parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

def noise_level_eval():
    args = grab_config()
    args.gpus = [0] # Force evaluation in a single gpu.
    args.results_dir = './results'
    args.num_tests = 10
    args.saved_model_type = 'baseline'
    seed_everything(42)
    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    trainer = Trainer.from_argparse_args(args, logger=logger, progress_bar_refresh_rate=0, auto_select_gpus=True)
    if not os.path.exists(os.path.join(args.results_dir, args.experiment_name)):
        os.mkdir(os.path.join(args.results_dir, args.experiment_name))

    all_results = []
    for test in range(args.num_tests):
        #Choose the appropriate model based on type, and load from checkpoint.
        basedir = '/work2/08002/gsmyrnis/maverick2/clip_experiments/code/Logs_Baselines/RN101_CLEAN_LINEAR/checkpoints/'
        checkpoint_path = os.path.join(basedir,os.listdir(basedir)[0])
        if args.saved_model_type == 'linear':
            saved_model = LinearProbe.load_from_checkpoint(checkpoint_path)
        elif args.saved_model_type == 'baseline':
            saved_model = Baseline.load_from_checkpoint(checkpoint_path)
        elif args.saved_model_type == 'zeroshot':
            saved_model = NoisyCLIPTesting(args, checkpoint_path)

        test_data = ImageNet100Test(args)
        results = trainer.test(model=saved_model, datamodule=test_data, verbose=False)
        all_results.extend(results)

    
    top1_accs = [x['test_top_1'] for x in all_results]
    top5_accs = [x['test_top_5'] for x in all_results]
    with open(os.path.join('./results', args.experiment_name, args.saved_model_type+'_fixed_res.out'), 'w+') as f:
        f.write('Top 1 mean\t{0:.4f}\n'.format(np.mean(top1_accs)))
        f.write('Top 1 std\t{0:.4f}\n'.format(np.std(top1_accs, ddof=1)))
        f.write('Top 1 stderr\t{0:.4f}\n'.format(np.std(top1_accs, ddof=1)/np.sqrt(args.num_tests)))
        f.write('Top 5 mean\t{0:.4f}\n'.format(np.mean(top5_accs)))
        f.write('Top 5 std\t{0:.4f}\n'.format(np.std(top5_accs, ddof=1)))
        f.write('Top 5 stderr\t{0:.4f}\n'.format(np.std(top5_accs, ddof=1)/np.sqrt(args.num_tests)))

if __name__ == "__main__":
    noise_level_eval()
