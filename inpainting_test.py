from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib notebook

import os
import argparse

import warnings
warnings.filterwarnings('ignore')

from deep_decoder.include import *
from PIL import Image
import PIL

import numpy as np
import torch
import torch.optim
from torch.autograd import Variable
from time import time

#Set up GPU usage
GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor

from utils import *
from deep_decoder import *
from torch.utils.data import DataLoader

from baselines import Baseline

def grab_config():
    """
    Given a filename, grab the corresponsing config file and return it 
    """
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

class ImageNetBaseTransformVal:
    """
    Small transform to just resize and crop a given image
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x)

args = grab_config()

#Load the saved model 
saved_model = Baseline.load_from_checkpoint(args.checkpoint_path)
saved_model = saved_model.encoder
saved_model = saved_model.to('cuda')
    
dataset = ImageNet100(root = '/tmp/ImageNet100',
                      split = 'val',
                      transform = ImageNetBaseTransformVal())

loader = DataLoader(dataset, batch_size=1, num_workers=12, pin_memory=True, shuffle=True)

#Set up the masking distortion and save the mask to be used in Deep Decoder
distortion = RandomMask(percent_missing=0.5, return_mask=True)
fake_tens = torch.zeros(1,3,224,224)
_, mask = distortion(fake_tens)

#Make the mask the correct size
mask = mask.view(224,224).unsqueeze(0)
mask = torch.cat(3*[mask]).unsqueeze(0)
mask = mask.to('cuda')

rnd = 500
numit = 5000
rn = 0.005
num_channels = [256]*5

top_1 = 0
top_5 = 0

#Set up the normalizer depending on what type of model we would like to use
if args.encoder == "clip":
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )
else:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

#Iterate through the entire validation set, perform inpainting on each image, then classify it
for i, batch in enumerate(loader):
    x, y = batch

    x = distortion(x)

    #initialize the DD network
    net = decodernw(3,num_channels_up=num_channels,upsample_first = True).type(dtype).to('cuda')

    x = x.to('cuda')

    #Perform inpainting
    mse_n, mse_t, ni, net = fit( num_channels=num_channels,
                            reg_noise_std=rn,
                            reg_noise_decayevery = rnd,
                            num_iter=numit,
                            LR=0.0025,
                            img_noisy_var=x,
                            net=net,
                            img_clean_var=x,
                            mask_var = mask,
                            find_best=True,
                            )

    out_img = net( ni.type(dtype) ).data
    out_img = normalize(out_img) #normalize the inpainted image
    
    logits = saved_model(out_img).squeeze()

    #Grab the top-1 and top-5 predictions from the network
    _, inds1 = torch.topk(logits, 1)
    _, inds5 = torch.topk(logits, 5)

    if y.item() in inds1:
        top_1 = top_1 + 1
        print("TOP 1")
    if y.item() in inds5:
        top_5 = top_5 + 1
        print("TOP 5")
    
    #If we have run through the entire dataset, save the final results
    if i >= 4999:
        top_1 = top_1 / i
        top_5 = top_5 / i

        results_file = os.path.join(args.results_dir, args.experiment_name + '.out')


        with open(results_file, 'w+') as f:
            f.write('Top 1\t{0:.4f}\n'.format(top_1))
            f.write('Top 5\t{0:.4f}\n'.format(top_5))
    
        break
    