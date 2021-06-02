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
from generative_inpainting_pytorch.model.networks import Generator
from generative_inpainting_pytorch.gen_utils.tools import normalize as gen_normalize

class BaselineDenoiseDistortVal:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For training, this class will apply a random crop and random horizontal flip as well.
    Explicitly saves the distortion as a class variable to use for fixed masks in validation transform if needed.
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

        if args.distortion == "randommask":
            distortion_denoise = RandomMaskGenerativeInpainting(percent_missing=convnoise(args.percent_missing, epoch))
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                distortion_denoise,
                normalize
            ])
        elif args.distortion == "gaussiannoise":
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

class RandomMaskGenerativeInpainting:
    """
    Torchvision transform that tries to perform generative inpainting on an image,
    which has some of its pixels erased.
    """
    def __init__(self, percent_missing):
        assert isinstance(percent_missing, float) or isinstance(percent_missing, list)
        self.percent_missing = percent_missing
        self.netG = Generator({'input_dim': 3, 'ngf': 32}, False, None)

    def __call__(self, image):
        h, w = image.shape[-2:]

        if isinstance(self.percent_missing, float):
            removed_num = int(h*w*self.percent_missing)
        else:
            removed_percent = np.random.uniform(self.percent_missing[0], self.percent_missing[1])
            removed_num = int(h*w*removed_percent)
        removed_secs = np.random.choice(h*w, removed_num, replace=False)
        mask = torch.ones(h*w)
        mask[removed_secs] = 0
        mask = 1.-mask
        mask = mask.view(h,w)
        mask = torch.cat(3*[mask.unsqueeze(dim=0)])

        image_norm = gen_normalize(image)
        print(image_norm.shape)
        print(mask.shape)
        image_missing_pixels = image_norm*(1.-mask)
        image1, image2, _ = self.netG(image_missing_pixels.unsqueeze(dim=0), mask.unsqueeze(dim=0))
        recon = image2* mask.unsqueeze(dim=0) + image_missing_pixels.unsqueeze(dim=0) * (1. - mask.unsqueeze(dim=0))

        # Return to proper range
        recon = recon.add_(1.).mul_(0.5)

        return recon.view(h,w)
