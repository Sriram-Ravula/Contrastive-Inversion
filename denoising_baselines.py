import numpy as np
import torch
import torchvision
from torchvision.transforms import Lambda
import os
import yaml
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets import ImageFolder
from utils import *

from skimage.restoration import denoise_nl_means, estimate_sigma

class BaselineDenoiseDistortVal:
    """
    Wrapper for the denoising transformation performed via NLMeans.
    """
    def __init__(self, args, epoch=None):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        if args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std, epoch), fixed=args.fixed_mask)
            denoise = NLMeansDenoising()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                distortion,
                denoise,
                normalize
            ])
        else:
            raise NotImplementedError('Baseline for this distortion not implemented.')

    def __call__(self, x):
        return self.transform(x)


class NLMeansDenoising:
    """
    Torchvision transform that performs non-local means denoising on a given image.
    Based on the skimage.denoise_nl_means function.
    """
    def __init__(self, sigma_given=None):
        self.sigma_given = sigma_given

    def __call__(self, x):
        img = x.permute(1,2,0).cpu().numpy()
        if self.sigma_given is None:
            sigma_est = np.mean(estimate_sigma(img, multichannel=True))
        else:
            sigma_est = self.sigma_given
        img_denoised = denoise_nl_means(img, sigma=sigma_est, fast_mode=True, multichannel=True)
        return torch.Tensor(img_denoised).to(x.device).permute(2,0,1)
