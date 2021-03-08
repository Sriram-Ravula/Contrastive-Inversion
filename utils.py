import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import os
import yaml
from torch.utils.data.dataset import Subset


class RandomMask(object):
    """
    Custom Torchvision transform meant to be used on image data with dimension (N, C, H, W).
    Mask an image with a random mask of missing pixels (salt and pepper noise)

    Args:
        percent_missing: percent of the pixels to mask
    """
    
    def __init__(self, percent_missing):
        assert isinstance(percent_missing, float)
        
        self.percent_missing = percent_missing
    
    def __call__(self, image):
        h, w = image.shape[-2:]
        
        removed_secs = np.random.choice(h*w, int(h*w*self.percent_missing))

        mask = torch.ones(h*w)
        mask[removed_secs] = 0
        
        return image*mask.view(h, w)

class SquareMask(object):
    """
    Custom Torchvision transform meant to be used on image data with dimension (N, C, H, W).
    Mask an image with a square mask of missing pixels
    
    Args:
        length: side length of the square masked area
        offset: {"center": center the square in the image,
                "random": perform a random vertical and horizontal offset of the square}
    """
    
    def __init__(self, length, offset="center"):
        viable_offsets = ["center", "random"]
        
        assert isinstance(offset, str)
        assert isinstance(length, int)
        
        assert offset in viable_offsets 
        
        self.offset = offset
        self.length = length
    
    def __call__(self, image):
        h, w = image.shape[-2:]
        
        assert (self.length < h and self.length < w)
        
        if self.offset == "random":
            #The random offsets define the center of the square region
            h_offset = np.random.choice(np.arange(self.length//2, h-(self.length//2)+1))
            w_offset = np.random.choice(np.arange(self.length//2, w-(self.length//2)+1))
            
            removed_secs_h = np.arange(h_offset-(self.length//2), h_offset+(self.length//2))
            removed_secs_w = np.arange(w_offset-(self.length//2), w_offset+(self.length//2))
            
            x, y = np.meshgrid(removed_secs_h, removed_secs_w)

            mask = torch.ones(h, w)
            mask[x, y] = 0
            
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
            
            return image*mask

class ImageNetSquareMask:
    """
    Torchvision composition of transforms to produce ImageNet images with the center square-masked
    """
    def __init__(self, mask_length = 50):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            SquareMask(length=mask_length, offset="center"),
            normalize
        ])
    
    def __call__(self, x):
        return self.transform(x)

class ImageNetSquareMaskVal:
    """
    Torchvision composition of transforms to produce ImageNet images with the center square-masked
    """
    def __init__(self, mask_length = 50):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            SquareMask(length=mask_length, offset="center"),
            normalize
        ])
    
    def __call__(self, x):
        return self.transform(x)

def img_grid(data):
    data = data.cpu()[0:64]

    grid = torchvision.utils.make_grid(data, nrow=8, normalize=True)

    return grid

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
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
    "Computes the Top-k accuracy (target is in the top k predictions)."
    input = input.topk(k=k, dim=-1)[1]
    targs = targs.unsqueeze(dim=-1).expand_as(input)
    
    return (input == targs).max(dim=-1)[0].float().sum()

def get_subset(dataset, filename):
    """
    Takes an original dataset and the name of a file with a subset of the original classes in the dataset.
    Returns the specified subset of the dataset and a dictionary of {old_label: new_label}.
    
    Arguments:
    dataset - the name of the dataset (currently only ImageNet supported)
    filename - the path to the text file containing the subset of data classes we want to keep
    
    Returns:
    ds_subset - the subset of the original dataset, keeping only samples with classes matching those in filename
    og_to_new_dict - dictionary of {old_label: new_label}
    """
    
    with open(filename) as f:
        subset_wnids = f.readlines()
    subset_wnids = [x.strip() for x in subset_wnids]
    
    wnid_idx = dataset.wnid_to_idx
    
    subset_og_classes = [wnid_idx[wnid] for wnid in subset_wnids]
    
    subset_img_indices = [i for i, label in enumerate(dataset.targets) if label in subset_og_classes]
    
    ds_subset = Subset(dataset, subset_img_indices)
    
    og_to_new_dict = {x: i for i, x in enumerate(subset_og_classes)}
    
    return ds_subset, og_to_new_dict #return the subset of the original dataset and the {old: new} class label dict

def map_classes(og_classes, remap):
    """
    Takes a list of classes for a batch of data and a map from old to new classes.
    Returns the re-mapped correct classes.
    
    Arguments: 
    og_classes - the tensor of original batch labels
    remap - the dictionary to remap classes {old_label: new_label}
    """
    
    x = og_classes.cpu().numpy()
    new_classes = torch.tensor([remap[i] for i in x])
    new_classes = new_classes.type_as(og_classes)
    
    return new_classes