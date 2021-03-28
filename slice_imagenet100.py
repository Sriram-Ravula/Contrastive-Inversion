#!/usr/bin/env python

import os
import sys
import shutil
from glob import glob

ORIG_IMAGENET_DIR = '../data/imagenet'
IMAGENET100_DIR = '../data/imagenet100'
IMAGENET100_CLASSES = 'imagenet100.txt'

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

  if __name__ == 'main':
      main()