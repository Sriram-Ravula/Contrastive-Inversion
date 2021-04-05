import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import os
import yaml
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets import ImageFolder


class RandomMask(object):
    """
    Custom Torchvision transform meant to be used on image data with dimension (N, C, H, W).
    Mask an image with a random mask of missing pixels (blacked out - values set to 0).

    Args:
        percent_missing: percent of the pixels to mask
        fixed: whether the mask is fixed for all images
    """

    def __init__(self, percent_missing, fixed=False):
        assert isinstance(percent_missing, float)

        self.percent_missing = percent_missing
        self.fixed = fixed
        self.mask = None

    def __call__(self, image):
        h, w = image.shape[-2:]
        if self.fixed and self.mask is not None:
            return image*self.mask.view(h,w)

        removed_secs = np.random.choice(h*w, int(h*w*self.percent_missing), replace=False)
        mask = torch.ones(h*w)
        mask[removed_secs] = 0
        if self.fixed:
            self.mask = mask

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

        noise = torch.randn((c, h, w)) * self.std

        if self.fixed:
            self.noise = noise

        return image + noise

class ImageNetBaseTransform:
    """
    Torchvision composition of transforms equivalent to the one required for CLIP clean images.
    Takes a set of arguments and alters the normalization constants depending on the model being used.
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


class ImageNetDistortTrain:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For training, this class will apply a random crop and random horizontal flip as well.
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

        if args.distortion == "squaremask":
            distortion = SquareMask(length=args.length, offset="center", fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=args.percent_missing, fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=args.std, fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            distortion,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNetDistortVal:
    """
    Torchvision composition of transforms to produce ImageNet images with a distortion.
    For validation, this class will always crop from the center of the image and NOT apply a random horizontal flip.
    """
    def __init__(self, args, fixed_distortion=None):
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
            distortion = SquareMask(length=args.length, offset="center", fixed = args.fixed_mask)
        elif args.distortion == "randommask":
            distortion = RandomMask(percent_missing=args.percent_missing, fixed = args.fixed_mask)
        elif args.distortion == "gaussiannoise":
            distortion = GaussianNoise(std=args.std, fixed=args.fixed_mask)
        elif args.distortion == "gaussianblur":
            distortion = transforms.GaussianBlur(kernel_size=args.kernel_size, sigma=args.sigma)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            distortion,
            normalize
        ])

    def __call__(self, x):
        return self.transform(x)

class ImageNet100(ImageFolder):
    """
    Dataset for ImageNet100. Majority of code taken from torchvision.datasets.ImageNet.
    NOT TESTED YET.
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

        #from the dataset fodler class, we inherit two properties
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
        self.idx_to_class = {idx: cls
                             for idx, clss in enumerate(self.classes)
                             for i, cls in enumerate(clss) if i is 0}


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
    cfg - a parseable dictionary of configuration options
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

def top_k_accuracy(input, targs, k=1):
    """
    Computes the Top-k accuracy (target is in the top k predictions).

    Arguments:
    input - an [N, num_classes] tensor with a probability distribution over classes at each index i=1:N
    targs - an [N] tensor with ground truth class label for each sample i=1:N
    k - the k for the top-k values to take from input

    Returns:
    top_k_accuracy - the top-k accuracy of inputs relative to targs
    """
    input = input.topk(k=k, dim=-1)[1]
    targs = targs.unsqueeze(dim=-1).expand_as(input)

    return (input == targs).max(dim=-1)[0].float().sum()

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

def map_classes(og_classes, remap):
    """
    Takes a list of classes for a batch of data and a map from old to new classes.
    Returns the re-mapped correct classes.

    Arguments:
    og_classes - the tensor of original batch labels
    remap - the dictionary to remap classes {old_label: new_label}

    Returns:
    new_classes - a tensor of the same type as og_classes, with the new classes for the data batch
    """

    x = og_classes.cpu().numpy()
    new_classes = torch.tensor([remap[i] for i in x])
    new_classes = new_classes.type_as(og_classes)

    return new_classes
