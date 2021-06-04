#!/usr/bin/env python

import os
import sys
import shutil
from glob import glob

DATA_SRC_ROOT = './ImageNet-C' #The path to the ImageNet-C dataset root - should contain 5 files of each distortion and meta.bin
IMAGENET100_DIR = './ImageNet100C' #The destination for the new ImageNet100-C folder
IMAGENET100_CLASSES = './imagenet100.txt' #the file with the wnid names of the classes in the imagenet subset
ZIP_PATH = './' #destination for the zipped ImageNet100C folder

def zip_imagenet100c():
    """
    Creates a data folder containing a 100-class subset of ImageNet, then creates a zipped copy of it
    """
    #First make sure the directory we are given is correct!
    if not os.path.isdir(DATA_SRC_ROOT):
        raise Exception("Bad filepath given")

    #create the destiantion directories if they don't exist
    if not os.path.isdir(IMAGENET100_DIR):
        os.mkdir(IMAGENET100_DIR)

    #grab the subset wnids for the 100 class-subset
    with open(IMAGENET100_CLASSES) as f:
        subset_wnids = f.readlines()
    subset_wnids = [x.strip() for x in subset_wnids] #list of the 100 WNIDs we grab

    #Grab the names of all of the folders inside the root data source
    #Structure is distortion/sub_distortion/level/wnids
    for distortion in os.listdir(DATA_SRC_ROOT):
        if distortion != "meta.bin":
            print(distortion)

        folder_path = os.path.join(DATA_SRC_ROOT, distortion)

        if not os.path.isdir(folder_path):
            continue

        for sub_distortion in os.listdir(folder_path):
            print(sub_distortion)

            subfolder_path = os.path.join(folder_path, sub_distortion)

            if not os.path.isdir(subfolder_path):
                continue

            for level in os.listdir(subfolder_path):
                print(level)

                level_path = os.path.join(subfolder_path, level)

                #grab the correcrt validation d9recotires
                for wnid in os.listdir(level_path):
                    wnid_path = os.path.join(level_path, wnid)

                    if not os.path.isdir(wnid_path):
                        continue

                    if wnid in subset_wnids:
                        dest_path = os.path.join(IMAGENET100_DIR, distortion, sub_distortion, level, wnid)

                        shutil.copytree(wnid_path, dest_path)

    #copy the metadata bin file
    meta_file = os.path.join(DATA_SRC_ROOT, 'meta.bin')
    meta_dest = os.path.join(IMAGENET100_DIR, 'meta.bin')

    shutil.copy(meta_file, meta_dest)

    #Zip the destinatio file
    shutil.make_archive(ZIP_PATH + '/ImageNet100C', 'tar', IMAGENET100_DIR)

if __name__ == '__main__':
    zip_imagenet100c()
