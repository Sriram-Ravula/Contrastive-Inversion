import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import os
import yaml
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset


def convnoise(x, epoch=None):
    """
    Function which adds a possibility of increasing noise levels.
    """
    if epoch is None:
        return x
    else:
        return 0.9*x*(1-np.exp(-epoch/5)) + 0.1*x

class RandomMask(object):
    """
    Custom Torchvision transform meant to be used on image data with dimension (N, C, H, W).
    Mask an image with a random mask of missing pixels (blacked out - values set to 0).

    Args:
        percent_missing: percent of the pixels to mask
        fixed: whether the mask is fixed for all images
    """

    def __init__(self, percent_missing, fixed=False, return_mask=False):
        assert isinstance(percent_missing, float) or isinstance(percent_missing, list)

        self.percent_missing = percent_missing
        self.fixed = fixed
        self.mask = None
        self.return_mask = return_mask

    def __call__(self, image):
        h, w = image.shape[-2:]
        if self.fixed and self.mask is not None:
            return image*self.mask.view(h,w)

        if isinstance(self.percent_missing, float):
            removed_num = int(h*w*self.percent_missing)
        else:
            removed_percent = np.random.uniform(self.percent_missing[0], self.percent_missing[1])
            removed_num = int(h*w*removed_percent)
        removed_secs = np.random.choice(h*w, removed_num, replace=False)
        mask = torch.ones(h*w)
        mask[removed_secs] = 0
        if self.fixed:
            self.mask = mask

        if self.return_mask:
            self.return_mask = False
            return image*mask.view(h, w), mask

        return image*mask.view(h, w)

class SquareMask(object):
    """
    Custom Torchvision transform meant to be used on image data with dimension (N, C, H, W).
    Mask an image with a square mask of missing pixels

    Args:
        length: side length of the square masked area
        offset: {"center": center the square in the image,
                "random": perform a random vertical and horizontal offset of the square}
        fixed: whether the mask is the same for all images (only useful with offset="random")
    """

    def __init__(self, length, offset="center", fixed=False):
        viable_offsets = ["center", "random"]

        assert isinstance(offset, str)
        assert isinstance(length, int)

        assert offset in viable_offsets

        self.offset = offset
        self.length = length
        self.fixed = fixed
        self.mask = None

    def __call__(self, image):
        h, w = image.shape[-2:]
        assert (self.length < h and self.length < w)

        if self.fixed and self.mask is not None:
            return image*self.mask

        if self.offset == "random":
            #The random offsets define the center of the square region
            h_offset = np.random.choice(np.arange(self.length//2, h-(self.length//2)+1))
            w_offset = np.random.choice(np.arange(self.length//2, w-(self.length//2)+1))

            removed_secs_h = np.arange(h_offset-(self.length//2), h_offset+(self.length//2))
            removed_secs_w = np.arange(w_offset-(self.length//2), w_offset+(self.length//2))

            x, y = np.meshgrid(removed_secs_h, removed_secs_w)

            mask = torch.ones(h, w)
            mask[x, y] = 0
            if self.fixed:
                self.mask = mask

            return image*mask

        elif self.offset == "center":
            #The center offsets here are in the middle of the image
            h_offset = h//2
            w_offset = w//2

            removed_secs_h = np.arange(h_offset-(self.length//2), h_offset+(self.length//2))
            removed_secs_w = np.arange(w_offset-(self.length//2), w_offset+(self.length//2))

            x, y = np.meshgrid(removed_secs_h, removed_secs_w)

            mask = torch.ones(h, w)
            mask[x, y] = 0
            if self.fixed:
                self.mask = mask

            return image*mask

class GaussianNoise(object):
    """
    Torchvision transform to add random Gaussian noise to an input image.

    Arguments:
        std - the standard deviation of the random noise to add to the image.
              Can be a list of [low, high] to choose uniformly randomly in [low, high]
        fixed - whether or not the noise we are adding is fixed for all images
    """
    def __init__(self, std, fixed=False):
        self.std = std
        self.fixed = fixed
        self.noise = None

    def __call__(self, image):
        c, h, w = image.shape[-3:]

        if self.fixed and self.noise is not None:
            return image + self.noise

        if isinstance(self.std, list):
            std = np.random.uniform(low=self.std[0], high=self.std[1])
            noise = torch.randn((c, h, w)) * std
        else:
            noise = torch.randn((c, h, w)) * self.std

        if self.fixed:
            self.noise = noise

        return image + noise

class GeneralBaseTransform:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds no distortion to the image.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class GeneralBaseTransformVal:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds no distortion to the image, and also performs no flipping.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class GeneralDistortTrain:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds the required distortion to the image, to be used for training.
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

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=convnoise(args.percent_missing, epoch), fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std, epoch), fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            distortion,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class GeneralDistortVal:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds the required distortion to the image, but no random flipping, to be used for validation.
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

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=convnoise(args.percent_missing, epoch), fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std, epoch), fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            distortion,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class GeneralDistortTrainMulti:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds a series of various distortion to the image.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            randjitter,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class GeneralDistortValMulti:
    """
    Transform to be used for CIFAR10, CIFAR100, and STL10 datasets.
    This adds a series of various distortion to the image, but no random flipping.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            randjitter,
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetBaseTransform:
    """
    Torchvision composition of transforms equivalent to the one required for CLIP clean images.
    Takes a set of arguments and alters the normalization constants depending on the model being used.
    Basic training transform with a random crop and horizointal flip.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetBaseTransformVal:
    """
    Torchvision composition of transforms equivalent to the one required for CLIP clean images.
    For validation, this class will always crop from the center of the image and NOT apply a random horizontal flip.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetBaseTrainContrastive:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For training, this class will apply a random crop and random horizontal flip as well.
    This explicitly returns a pair of images (clean, clean) with the same random crop and flip.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        x_clean = self.transform(x)
        x_clean_copy = x_clean.clone()

        return x_clean, x_clean_copy

class ImageNetBaseTrainContrastiveDecoupled:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For training, this class will apply a random crop and random horizontal flip as well.
    This explicitly returns a pair of images (clean, clean) with different random crop and flip.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)

        return x_1, x_2

class ImageNetDistortTrainContrastive:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For training, this class will apply a random crop and random horizontal flip as well.
    This explicitly returns a pair of images (clean, noisy) with the same random crop and flip for both.
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
        self.normalize = normalize

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=convnoise(args.percent_missing,epoch), fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std,epoch), fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)
        self.distortion = distortion

        self.transform_common = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        x_temp = self.transform_common(x)
        x_clean = self.normalize(x_temp)
        x_noisy = self.normalize(self.distortion(x_temp))
        return x_clean, x_noisy

class ImageNetDistortValContrastive:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For validation, there is a deterministic resizing and center crop.
    This explicitly returns a pair of images (clean, noisy).
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.normalize = normalize

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=args.percent_missing, fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=args.std, fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)
        self.distortion = distortion

        self.transform_common = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        x_temp = self.transform_common(x)
        x_clean = self.normalize(x_temp)
        x_noisy = self.normalize(self.distortion(x_temp))
        return x_clean, x_noisy


class ImageNetDistortTrain:
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

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=convnoise(args.percent_missing, epoch), fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std, epoch), fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            distortion,
            normalize
        ])
        self.distortion = distortion

    def __call__(self, x):
        return self.transform(x)

class ImageNetDistortVal:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For validation, this class will always crop from the center of the image and NOT apply a random horizontal flip.
    Can pass a fixed distortion from a previously-initialized training distortive transform.
    """
    def __init__(self, args, fixed_distortion=None, epoch=None):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        if fixed_distortion is not None:
            distortion = fixed_distortion
        elif args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset=args.offset, fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=convnoise(args.percent_missing, epoch), fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=convnoise(args.std, epoch), fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            distortion,
            normalize
        ])
        self.distortion = distortion

    def __call__(self, x):
        return self.transform(x)

class ImageNetDistortTrainMulti:
    """
    Applies a series of transforms to an image:
    Random Crop to 224x224, Random horizontal clip p=0.5,
    Random color jitter p=0.8 (max adjustment for: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    Random Gaussian Blur p=0.1 (kernel 23x23, std unformly random in [0.1, 0.2]),
    Random Gaussian Noise p=0.2 (std uniformly random in [0.1, 0.3]),
    Random Pixel Mask p=0.3 (percentage masked uniformly random in [0.5, 0.95]).

    Returns a distorted image.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        randcrop = transforms.RandomResizedCrop(224)

        randflip = transforms.RandomHorizontalFlip()

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform = transforms.Compose([
            randcrop,
            randflip,
            randjitter,
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetDistortValMulti:
    """
    Applies a series of transforms to an image:
    Resize to 256x256, Center crop to 224x224,
    Random color jitter p=0.8 (max adjustment for: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    Random Gaussian Blur p=0.1 (kernel 23x23, std unformly random in [0.1, 0.2]),
    Random Gaussian Noise p=0.2 (std uniformly random in [0.1, 0.3]),
    Random Pixel Mask p=0.3 (percentage masked uniformly random in [0.5, 0.95]).

    Returns a distorted image.
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            randjitter,
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetDistortTrainMultiContrastive:
    """
    Applies a distortive transform to one copy of an image and a simple random crop and flip to another copy.
    Transformations are chosen based on mode:

    Mode 1:
    Random distortions on the distorted copy are:
    Random Crop to 224x224, Random horizontal clip p=0.5,
    Random color jitter p=0.5 (max adjustment for: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    Random Gaussian Blur p=0.4 (kernel 23x23, std unformly random in [1, 5]),
    Random Gaussian Noise p=0.4 (std uniformly random in [0.1, 0.5]),
    Random Pixel Mask p=0.1 (percentage masked uniformly random in [0.25, 0.5]).

    Mode 2:
    Distortions are the same as in mode 1, but at most one is applied each time.
    With p=0.6, one distortion is applied, chosen with probability proportional to the ones above.
    With p=0.4, no distortion is applied.

    Mode 1:
    Random distortions on the distorted copy are:
    Random Gaussian Blur p=0.8 (kernel 23x23, std unformly random in [1, 5]),
    Random Gaussian Noise p=0.8 (std uniformly random in [0.05, 0.5]),

    Mode 2:
    Distortions are the same as in mode 3, but at most one is applied each time.
    With p=0.6, one distortion is applied, chosen with probability proportional to the ones above.
    With p=0.4, no distortion is applied.

    Returns a pair of images (clean, distorted)
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        randcrop = transforms.RandomResizedCrop(224)
        randflip = transforms.RandomHorizontalFlip()


        self.transform_common = transforms.Compose([
            randcrop,
            randflip
        ])

        self.transform_clean = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if not args.mode or args.mode == 1:
            jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            randjitter = transforms.RandomApply([jitter], p=0.5)

            blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
            randblur = transforms.RandomApply([blur], p=0.4)

            noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
            randnoise = transforms.RandomApply([noise], p=0.4)

            mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
            randmask = transforms.RandomApply([mask], p=0.1)

            self.transform_distort = transforms.Compose([
                randjitter,
                transforms.ToTensor(),
                randblur,
                randnoise,
                randmask,
                normalize
            ])
        elif args.mode == 2:
            rng = torch.rand(1).item()
            if rng < 0.6*0.5/1.4:
                distort = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            elif rng < 0.6*0.9/1.4:
                distort = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
            elif rng < 0.6*1.3/1.4:
                distort = GaussianNoise(std=[0.1, 0.5], fixed=False)
            elif rng < 0.6:
                distort = RandomMask(percent_missing=[0.25, 0.50], fixed = False)

            if rng < 0.6*0.5/1.4:
                self.transform_distort = transforms.Compose([
                    distort,
                    transforms.ToTensor(),
                    normalize
                ])
            elif rng < 0.6:
                self.transform_distort = transforms.Compose([
                    transforms.ToTensor(),
                    distort,
                    normalize
                ])
            else:
                self.transform_distort = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])
        elif args.mode == 3:
            blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
            randblur = transforms.RandomApply([blur], p=0.8)

            noise = GaussianNoise(std=[0.05, 0.5], fixed=False)
            randnoise = transforms.RandomApply([noise], p=0.8)

            self.transform_distort = transforms.Compose([
                transforms.ToTensor(),
                randblur,
                randnoise,
                normalize
            ])
        elif args.mode == 4:
            rng = torch.rand(1).item()
            if rng < 0.6*0.8/1.6:
                distort = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
            elif rng < 0.6:
                distort = GaussianNoise(std=[0.1, 0.5], fixed=False)

            if rng < 0.6:
                self.transform_distort = transforms.Compose([
                    transforms.ToTensor(),
                    distort,
                    normalize
                ])
            else:
                self.transform_distort = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])

    def __call__(self, x):
        x_temp = self.transform_common(x)
        x_clean = self.transform_clean(x_temp)
        x_noisy = self.transform_distort(x_temp)

        return x_clean, x_noisy

class ImageNetDistortValMultiContrastive:
    """
    Applies a distortive transform to one copy of an image and a simple crop to another copy.

    Random distortions on the distorted copy are:
    Resize to 256x256, Center crop to 224x224,
    Random color jitter p=0.8 (max adjustment for: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    Random Gaussian Blur p=0.1 (kernel 23x23, std unformly random in [0.1, 0.2]),
    Random Gaussian Noise p=0.2 (std uniformly random in [0.1, 0.3]),
    Random Pixel Mask p=0.3 (percentage masked uniformly random in [0.5, 0.95]).

    Returns a pair of images (clean, distorted)
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        randcrop = transforms.RandomResizedCrop(224)

        randflip = transforms.RandomHorizontalFlip()

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform_common = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

        self.transform_clean = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.transform_distort = transforms.Compose([
            randjitter,
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        x_temp = self.transform_common(x)
        x_clean = self.transform_clean(x_temp)
        x_noisy = self.transform_distort(x_temp)

        return x_clean, x_noisy

class ImageNetDistortTrainMultiContrastiveDecoupled:
    """
    Applies a distortive transform to one copy of an image and a simple random crop and flip to another copy.

    Random distortions on the distorted copy are:
    Random Crop to 224x224, Random horizontal clip p=0.5,
    Random color jitter p=0.8 (max adjustment for: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    Random Gaussian Blur p=0.1 (kernel 23x23, std unformly random in [0.1, 0.2]),
    Random Gaussian Noise p=0.2 (std uniformly random in [0.1, 0.3]),
    Random Pixel Mask p=0.3 (percentage masked uniformly random in [0.5, 0.95]).

    Returns a pair of images (clean, distorted)
    """
    def __init__(self, args):
        if args.encoder == "clip":
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        randcrop = transforms.RandomResizedCrop(224)

        randflip = transforms.RandomHorizontalFlip()

        jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        randjitter = transforms.RandomApply([jitter], p=0.5)

        blur = transforms.GaussianBlur(kernel_size=23, sigma=[1, 5])
        randblur = transforms.RandomApply([blur], p=0.4)

        noise = GaussianNoise(std=[0.1, 0.5], fixed=False)
        randnoise = transforms.RandomApply([noise], p=0.4)

        mask = RandomMask(percent_missing=[0.25, 0.50], fixed = False)
        randmask = transforms.RandomApply([mask], p=0.1)

        self.transform_common = transforms.Compose([
            randcrop,
            randflip
        ])

        self.transform_clean = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.transform_distort = transforms.Compose([
            randjitter,
            transforms.ToTensor(),
            randblur,
            randnoise,
            randmask,
            normalize
        ])

    def __call__(self, x):
        x_clean = self.transform_clean(self.transform_common(x))
        x_noisy = self.transform_distort(self.transform_common(x))

        return x_clean, x_noisy

class ImageNet100(ImageFolder):
    """
    Dataset for ImageNet100. Majority of code taken from torchvision.datasets.ImageNet.
    Works in a similar function and has similar semantics to the original class.
    """
    def __init__(self, root, split, transform=None, **kwargs):
        #checking stuff
        root = os.path.expanduser(root)
        if split != 'train' and split != 'val':
            raise ValueError('Split should be train or val.')

        #contains our desired {wnid: class} dictionary
        META_FILE = "meta.bin"

        #initialize parameters from DatasetFolder
        super(ImageNet100, self).__init__(os.path.join(root, split), **kwargs)
        self.root = root
        self.split = split
        self.transform = transform

        #from the dataset folder class, we inherit two properties
        #self.classes is a list of class names based on the folders present in our subset - actually wnids!
        #self.class_to_idx is a dict {wnid: wnid_index} where wnid_index is a number from 0 to 99

        #Load the {wnid: class_name} dictionary from meta.bin
        wnid_to_classes = torch.load(os.path.join(self.root, META_FILE))[0]
        self.wnids = self.classes #current self.classes is actually wnids!
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids] #get the actual class names (e.g. "bird")
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        #create a dictionary of UNIQUE {index: class values} where the class is the simplest form of the wnid (e.g. common name and not scientific name)
        #this is for CLIP zero-shot classification using the simplest class name
        self.idx_to_class = {idx: cls
                             for idx, clss in enumerate(self.classes)
                             for i, cls in enumerate(clss) if i == 0}


class ImageNet100OOD(ImageFolder):
    """
    Dataset for ImageNet100. Majority of code taken from torchvision.datasets.ImageNet.
    Works in a similar function and has similar semantics to the original class.
    """
    def __init__(self, root, split, transform=None, **kwargs):
        #checking stuff
        root = os.path.expanduser(root)
        if split != 'val':
            raise ValueError('Split should be val.')

        #contains our desired {wnid: class} dictionary
        META_FILE = "meta.bin"

        #initialize parameters from DatasetFolder
        super(ImageNet100OOD, self).__init__(os.path.join(root, split), **kwargs)
        self.root = root
        self.split = split
        self.transform = transform

        #from the dataset folder class, we inherit two properties
        #self.classes is a list of class names based on the folders present in our subset - actually wnids!
        #self.class_to_idx is a dict {wnid: wnid_index} where wnid_index is a number from 0 to 99

        #Load the {wnid: class_name} dictionary from meta.bin
        wnid_to_classes = torch.load(os.path.join(self.root, META_FILE))[0]
        self.wnids = self.classes #current self.classes is actually wnids!
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids] #get the actual class names (e.g. "bird")
        self.class_to_idx = {
            ('cup',): 53,
            ('Dungeness crab', 'Cancer magister'): 8,
            ('mountain bike','all-terrain bike', 'off-roader'): 68,
            ('wood rabbit', 'cottontail', 'cottontail rabbit'): 40,
            ('French bulldog',): 21
        }
        for key in self.wnid_to_idx.keys():
            self.wnid_to_idx[key] = self.class_to_idx[wnid_to_classes[key]]
        print(self.wnid_to_idx)
        self.samples = make_dataset(os.path.join(root, split), self.wnid_to_idx, extensions=('jpeg',))
        self.targets = [s[1] for s in self.samples]

class ImageNet100C(ImageFolder):
    """
    Dataset for ImageNet100C. Majority of code taken from torchvision.datasets.ImageNet.
    Works in a similar function and has similar semantics to the original class.
    """
    def __init__(self, root, distortion, sub_distortion, level, split='val', transform=None, **kwargs):
        #checking stuff
        root = os.path.expanduser(root)
        data_root = os.path.join(root, distortion, sub_distortion, level)

        if not os.path.isdir(data_root):
            raise Exception("BAD PATH! DATASET DOES NOT EXIST!")

        #contains our desired {wnid: class} dictionary
        META_FILE = "meta.bin"

        #initialize parameters from DatasetFolder
        super(ImageNet100C, self).__init__(data_root, **kwargs)
        self.root = data_root
        self.split = split
        self.transform = transform

        #from the dataset folder class, we inherit two properties
        #self.classes is a list of class names based on the folders present in our subset - actually wnids!
        #self.class_to_idx is a dict {wnid: wnid_index} where wnid_index is a number from 0 to 99

        #Load the {wnid: class_name} dictionary from meta.bin
        wnid_to_classes = torch.load(os.path.join(root, META_FILE))[0]
        self.wnids = self.classes #current self.classes is actually wnids!
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids] #get the actual class names (e.g. "bird")
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        #create a dictionary of UNIQUE {index: class values} where the class is the simplest form of the wnid (e.g. common name and not scientific name)
        #this is for CLIP zero-shot classification using the simplest class name
        self.idx_to_class = {idx: cls
                             for idx, clss in enumerate(self.classes)
                             for i, cls in enumerate(clss) if i == 0}

    @property
    def split_folder(self) -> str:
        return self.root

def img_grid(data):
    """
    Creates an 8x8 grid out of given batch of images.

    Arguments:
        data - an [N, 3, H, W] tensor of images, where N >= 64. Does not have to be normalized to [0, 1] or [-1, 1]

    Returns:
        grid - an 8x8 grid of images
    """
    data = data.cpu()[0:64]

    grid = torchvision.utils.make_grid(data, nrow=8, normalize=True)

    return grid

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files.

    Arguments:
        config_file - a .yaml file with a dictionary of configuration parameters

    Returns:
        cfg - a parseable object of configuration options
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def get_subset(dataset, filename, return_class_labels=False):
    """
    Takes an original dataset and the name of a file with a subset of the original classes in the dataset.
    Returns the specified subset of the dataset and a dictionary of {old_label: new_label} (as well as these
    labels as text, if desired).

    Arguments:
        dataset - the name of the dataset (currently only ImageNet supported)
        filename - the path to the text file containing the subset of data classes we want to keep
        return_class_labels - whether to return the class labels as text or not

    Returns:
        ds_subset - the subset of the original dataset, keeping only samples with classes matching those in filename
        og_to_new_dict - dictionary of {old_label: new_label}
        text_labels - list of labels as text (only returned if return_class_labels=True)
    """

    with open(filename) as f:
        subset_wnids = f.readlines()
    subset_wnids = [x.strip() for x in subset_wnids]

    wnid_idx = dataset.wnid_to_idx

    subset_og_classes = [wnid_idx[wnid] for wnid in subset_wnids]

    subset_img_indices = [i for i, label in enumerate(dataset.targets) if label in subset_og_classes]

    ds_subset = Subset(dataset, subset_img_indices)

    og_to_new_dict = {x: i for i, x in enumerate(subset_og_classes)}

    if not return_class_labels:
        return ds_subset, og_to_new_dict #return the subset of the original dataset and the {old: new} class label dict
    else:
        idxs = [dataset.wnid_to_idx[wnid] for wnid in subset_wnids]
        class_list = list(dataset.class_to_idx.keys())
        idx_list = list(dataset.class_to_idx.values())
        text_labels = []
        for idx in idxs:
            pos = idx_list.index(idx)
            text_labels.append(class_list[pos])
        return ds_subset, og_to_new_dict, text_labels

def few_shot_dataset(dataset, num_samples, n_classes=100):
    """
    A method to randomly subset the classes in a given dataset, with an equal number of samples per class in the subset.
    Used mainly for few-shot linear evaluation.

    Args:
        dataset - the torch dataset to subset
        num_samples - the number of samples to keep in each class
        n_classes - the number of classes in the given dataset

    Returns:
        few_shot_subset - the dataset with the few samples per class kept
    """

    subset_img_indices = []

    for n in range(n_classes):
        #grab the indices of all the images in the current class
        original_img_inds = [i for i, label in enumerate(dataset.targets) if label == n]

        #grab num_samples random image indices from the class
        random_img_subset = np.random.choice(original_img_inds, size=num_samples, replace=False)

        #add this random subset of images from the same class to our master subset
        subset_img_indices.extend(random_img_subset)

    few_shot_subset = Subset(dataset, subset_img_indices)

    return few_shot_subset
