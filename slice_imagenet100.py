#!/usr/bin/env python

import os
import sys
import shutil
from glob import glob

ORIG_IMAGENET_DIR = './ILSVRC/Data/CLS-LOC' #The path to the ImageNet dataset - should point to the CLS-LOC folder root
IMAGENET100_DIR = './ImageNet100' #The destination for the new ImageNet100 folder
IMAGENET100_CLASSES = 'imagenet100.txt' #the file with the wnid names of the classes in the imagenet subset
ZIP_PATH = './'

def zip_imagenet100():
    """
    Creates a data folder containing a 100-class subset of ImageNet, then creates a zipped copy of it
    """
    #First make sure the directory we are given is correct!
    if not os.path.isdir(ORIG_IMAGENET_DIR):
        raise Exception("Bad filepath given")

    #train and val directories to place the new image classes
    new_train_root = os.path.join(IMAGENET100_DIR, 'train')
    new_val_root = os.path.join(IMAGENET100_DIR, 'val')

    #create the destiantion directories if they don't exist
    if not os.path.isdir(IMAGENET100_DIR):
        os.mkdir(IMAGENET100_DIR)
        os.mkdir(new_train_root)
        os.mkdir(new_val_root)

    class_path = os.path.join(ORIG_IMAGENET_DIR, IMAGENET100_CLASSES)

    #grab the subset wnids for the 100 class-subset
    with open(class_path) as f:
        subset_wnids = f.readlines()
    subset_wnids = [x.strip() for x in subset_wnids] #list of the 100 WNIDs we grab

    #paths to original train and val
    train_path = os.path.join(ORIG_IMAGENET_DIR, 'train')
    val_path = os.path.join(ORIG_IMAGENET_DIR, 'val')

    #grab the correct training direcotries
    for folder in os.listdir(train_path):
        folder_path = os.path.join(train_path, folder)

        if not os.path.isdir(folder_path):
            continue

        if folder in subset_wnids:
            dest_path = os.path.join(new_train_root, folder)
            shutil.copytree(folder_path, dest_path)


    #grab the correcrt validation d9recotires
    for folder in os.listdir(val_path):
        folder_path = os.path.join(val_path, folder)

        if not os.path.isdir(folder_path):
            continue

        if folder in subset_wnids:
            dest_path = os.path.join(new_val_root, folder)
            shutil.copytree(folder_path, dest_path)

    #copy the metadata bin file
    meta_file = os.path.join(ORIG_IMAGENET_DIR, 'meta.bin')
    meta_dest = os.path.join(IMAGENET100_DIR, 'meta.bin')

    shutil.copy(meta_file, meta_dest)

    #Zip the destinatio file
    shutil.make_archive(ZIP_PATH + '/ImageNet100', 'tar', IMAGENET100_DIR)

if __name__ == '__main__':
    zip_imagenet100()
