import os
import clip
import torch
from utils import RandomMask, SquareMask
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur
from PIL import Image

def get_features(model, dataset, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=256)):
            features = model.encode_image(images.to(device))

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def get_test_transform(trans, args, n_px):

    if trans == "Random":
        percent_missing = args[0]

        test_preprocess = Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            RandomMask(percent_missing),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    elif trans == "Square":
        square_len = args[0]
        square_offset = "random"

        test_preprocess = Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            SquareMask(square_len, square_offset),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    elif trans == "Blur":
        blur_size = args[0]
        blur_std = args[1]

        test_preprocess = Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            GaussianBlur(blur_size, blur_std),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    
    return test_preprocess

def get_clf(train_features, train_labels):
    parameters = {'C':[0.01, 0.03, 0.05, 0.075, 0.1, 0.3, 0.5, 0.75, 1]}
    #parameters = {'C':[0.01]}

    logistic = LogisticRegression(random_state=0, max_iter=1000)

    clf = GridSearchCV(logistic, parameters, n_jobs=-1, verbose=2, cv=10)
    clf.fit(train_features, train_labels)

    print("Best Parameters: ") 
    print(clf.best_params_)

    return clf

def main(debug=True, data="CIFAR100"):
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

    if device=="cuda":
        torch.cuda.empty_cache()

    root = os.path.expanduser("~/.cache")

    models = ['RN50', 'ViT-B/32']

    trans_types = ["None", "Random", "Blur", "Square"]

    pct_missings = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    #pct_missings = [0.01]

    sq_lens = [10, 20, 30, 50, 75, 100, 150, 200]
    #sq_lens = [20]

    blur_size = 11
    blur_stds = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    #blur_stds = [0.1]

    for m in models:
        if debug:
           print("LOADING MODEL: " + m) 
           start = time.time()

        model, train_preprocess = clip.load(m, device)
        n_px = model.input_resolution.item()

        if debug:
           print("\nELAPSED TIME: ", str(time.time() - start)) 

        if debug:
           print("\nCALCULATING TRAINING FEATURES") 
           start = time.time()

        train = CIFAR100(root, download=False, train=True, transform=train_preprocess)
        train_features, train_labels = get_features(model, train, device)

        if debug:
           print("\nELAPSED TIME: ", str(time.time() - start)) 

        if debug:
            print("\nPERFORMING CROSS-VALIDATION TO FIND BEST LOGISTIC REGRESSION MODEL")
            start = time.time()

        clf = get_clf(train_features, train_labels)

        if debug:
           print("\nELAPSED TIME: ", str(time.time() - start)) 

        for trans in trans_types:
            name_str = m
            if name_str == 'ViT-B/32':
                name_str = 'ViT'
            name_str += trans

            if trans=="None":
                accuracies = []

                if debug:
                    print("\n====Baseline Models (Clean Data)====")
                    start = time.time()

                test_preprocess = train_preprocess
                test = CIFAR100(root, download=False, train=False, transform=test_preprocess)
                test_features, test_labels = get_features(model, test, device)

                predictions = clf.predict(test_features)
                accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
                print(f"Accuracy = {accuracy:.3f}")

                accuracies.append(accuracy)

                np.savetxt(os.getcwd()+"/results/"+data+"/"+name_str, np.array(accuracies))

                if debug:
                    print("\nELAPSED TIME: ", str(time.time() - start)) 
            
            elif trans=="Random":
                if debug:
                    print("\n====Random Masking====")
                
                accuracies = []

                for pct in pct_missings:
                    if debug:
                        print("PCT MISSING: ", str(pct))
                        start = time.time()

                    test_preprocess = get_test_transform(trans, [pct], n_px)
                    test = CIFAR100(root, download=False, train=False, transform=test_preprocess)
                    test_features, test_labels = get_features(model, test, device)

                    predictions = clf.predict(test_features)
                    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
                    print(f"Accuracy = {accuracy:.3f}")

                    accuracies.append(accuracy)

                    if debug:
                        print("\nELAPSED TIME: ", str(time.time() - start)) 
                
                np.savetxt(os.getcwd()+"/results/"+data+"/"+name_str, np.array(accuracies))

            elif trans=="Square":
                if debug:
                    print("\n====Square Masking====")
                
                accuracies = []

                for sq_len in sq_lens:
                    if debug:
                        print("SQUARE LENGTH: ", str(sq_len))
                        start = time.time()

                    test_preprocess = get_test_transform(trans, [sq_len], n_px)
                    test = CIFAR100(root, download=False, train=False, transform=test_preprocess)
                    test_features, test_labels = get_features(model, test, device)

                    predictions = clf.predict(test_features)
                    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
                    print(f"Accuracy = {accuracy:.3f}")

                    accuracies.append(accuracy)

                    if debug:
                        print("\nELAPSED TIME: ", str(time.time() - start)) 
                
                np.savetxt(os.getcwd()+"/results/"+data+"/"+name_str, np.array(accuracies))

            elif trans=="Blur":
                if debug:
                    print("\n====Blurring====")
                
                accuracies = []
                
                for std in blur_stds:
                    if debug:
                        print("BLUR STD: ", str(std))
                        start = time.time()

                    test_preprocess = get_test_transform(trans, [blur_size, std], n_px)
                    test = CIFAR100(root, download=False, train=False, transform=test_preprocess)
                    test_features, test_labels = get_features(model, test, device)

                    predictions = clf.predict(test_features)
                    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
                    print(f"Accuracy = {accuracy:.3f}")

                    accuracies.append(accuracy)

                    if debug:
                        print("\nELAPSED TIME: ", str(time.time() - start)) 
                
                np.savetxt(os.getcwd()+"/results/"+data+"/"+name_str, np.array(accuracies))

    return

if __name__ == "__main__":
    main()