#!/usr/bin/env python

import os
import sys
import shutil
from glob import glob

ORIG_IMAGENET_DIR = '/home/sriram/Projects/Datasets/Imagenet/ILSVRC/Data/CLS-LOC'
IMAGENET100_DIR = '/home/sriram/Projects/Datasets/ImageNet100'
IMAGENET100_CLASSES = 'imagenet100.txt'
ZIP_PATH = '/home/sriram/Projects/Datasets'

def zip_imagenet100():
    """
    Creates a zip file of the 100 class split of imagenet train and validation data
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
    

def main():
    if not os.path.isdir(IMAGENET100_DIR):
        os.mkdir(IMAGENET100_DIR)

    classes = [line.strip() for line in open(IMAGENET100_CLASSES, 'r')]

    train_orig_dir = os.path.join(ORIG_IMAGENET_DIR, 'train')
    val_orig_dir = os.path.join(ORIG_IMAGENET_DIR, 'val')

    train_100_dir = os.path.join(IMAGENET100_DIR, 'train')
    val_100_dir = os.path.join(IMAGENET100_DIR, 'val')

    for name in os.listdir(train_orig_dir):
        if not os.path.isdir(name):
            continue

        if name in classes:
            shutil.copytree(os.path.join(train_orig_dir, name), os.path.join(train_100_dir, name))

    for name in os.listdir(val_orig_dir):
        if not os.path.isdir(name):
            continue

        if name in classes:
            shutil.copytree(os.path.join(val_orig_dir, name), os.path.join(val_100_dir, name))

if __name__ == '__main__':
    zip_imagenet100()