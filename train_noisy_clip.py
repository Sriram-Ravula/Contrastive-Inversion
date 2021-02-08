#!/usr/bin/env python

import numpy as np
import os
import sys
import pickle
import time
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import clip
from utils import RandomMask, SquareMask
from noisy_clip import ContrastiveUnsupervisedDataset, ModifiedCLIP
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

def get_features(model, dataset, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=512, shuffle=False)):
            features = model.encode_image(images.to(device))

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).to(device), torch.cat(all_labels).to(device)

def main(debug=True, data="CIFAR10", batch_size = 8, saved_embeddings = True):
    # Keep these guys here for cmd arguments eventually
    # debug = None
    # if len(args) > 1:
    #     debug = args[1]
    # if debug == "True" or debug == "true" or debug == "y" or debug == "Y":
    #     debug = True
    #     print("Debug: True")
    # else:
    #     debug = False
    #     print("Debug: False")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    if device=="cuda":
        torch.cuda.empty_cache()

    root = os.path.expanduser("~/.cache") 

    # models = ['RN50', 'ViT-B/32']
    models = ['ViT-B/32']

    # trans_types = ["None", "Random", "Blur", "Square"]
    trans_types = ['Random']

    # pct_missings = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    pct_missings = [0.01]

    sq_lens = [10, 20, 30, 50, 75, 100, 150, 200]
    # sq_lens = [20]

    blur_size = 11
    blur_stds = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    #blur_stds = [0.1]

    for m in models:
        if debug:
           print("LOADING MODEL: " + m)
           start = time.time()

        if debug:
           print("\nELAPSED TIME: ", str(time.time() - start))

        if debug:
            print("\nPERFORMING CROSS-VALIDATION TO FIND BEST LOGISTIC REGRESSION MODEL")
            start = time.time()

        if debug:
           print("\nELAPSED TIME: ", str(time.time() - start))

        for trans in trans_types:

            name_str = m
            if name_str == 'ViT-B/32':
                name_str = 'ViT'
            name_str += trans

            if trans=="Random":
                if debug:
                    print("\n====Random Masking====")

                accuracies = []

                for pct in pct_missings:
                    if debug:
                        print("PCT MISSING: ", str(pct))
                        start = time.time()

                    if not saved_embeddings:
                        with torch.no_grad():
                            baseclip, train_preprocess = clip.load(m, device)
                            n_px = baseclip.input_resolution.item()
                            train_base = CIFAR100(root, download=True, train=True, transform=train_preprocess)
                            clean_embeddings, _ = get_features(baseclip, train_base, device)
                            CIFAR10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse','ship','truck']
                            text_embeddings = baseclip.encode_text(clip.tokenize(['a picture of a '+label for label in CIFAR10_labels]).to(device))
                            # clean_embeddings = clean_embeddings.requires_grad=False
                            # text_embeddings = text_embeddings.requires_grad=False
                            pickle.dump((clean_embeddings, text_embeddings), open('embeddings.pkl','wb'))
                        del baseclip
                    else:
                        clean_embeddings, text_embeddings = pickle.load(open('embeddings.pkl','rb'))
                        _, train_preprocess = clip.load(m, device)
                        train_base = CIFAR10(root, download=True, train=True, transform=train_preprocess)

                    
                    train_contrastive = ContrastiveUnsupervisedDataset(train_base, clean_embeddings, transform_noisy=RandomMask(pct))
                    train_dl = DataLoader(train_contrastive, batch_size=batch_size, shuffle=True)

                    new_model = ModifiedCLIP(text_embeddings, device=device)
                    new_model.fit(train_dl)

                    test_dataset = CIFAR10(root, download=True, train=False, transform=Compose([train_preprocess, RandomMask(pct)]))
                    
                    test_dl = DataLoader(test_dataset, batch_size=batch_size)
                    acc = new_model.score(test_dl)

                    accuracies.append(acc)

                    if debug:
                        print("\nELAPSED TIME: ", str(time.time() - start))

                if not os.path.exists(os.path.join(os.getcwd(),'/results_noisy/')):
                    os.mkdir(os.path.join(os.getcwd(),'/results_noisy/'))
                if not os.path.exists(os.path.join(os.getcwd(),'/results_noisy/', data)):
                    os.mkdir(os.path.join(os.getcwd(),'/results_noisy/', data))
                np.savetxt('results_noisy_CIFAR100', np.array(accuracies))

            elif trans=="Square":
                raise NotImplementedError()

            elif trans=="Blur":
                raise NotImplementedError()

    return

if __name__ == "__main__":
    main()
