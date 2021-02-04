import numpy as np
import torch

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