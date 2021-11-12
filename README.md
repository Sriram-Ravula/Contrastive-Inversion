# Inverse Problems Leveraging Pre-trained Contrastive Representations

This repo contains the official code for the paper [Inverse Problems Leveraging Pre-trained Contrastive Representations](https://arxiv.org/abs/2110.07439).

Authors: [Sriram Ravula](https://www.sriramravula.com), [Georgios Smyrnis](https://georgiossmyrnis.github.io/), [Matt Jordan](https://www.cs.utexas.edu/~mjordan/), and [Alexandros G. Dimakis](https://users.ece.utexas.edu/~dimakis/) from The University of Texas at Austin.

## Requirements

The file ```requirements.txt``` provides the environment specifications for running our code. You can install them using ```pip install -r requirements.txt```. You also need the following files from the OpenAI CLIP repository (https://github.com/openai/CLIP), along with their respective requirements:
- ```model.py```
- ```clip.py```
- ```simple_tokenizer.py```
- ```bpe_simple_vocab_16e6.txt.gz```

For simplicity, we have copied these required files into our own code in the folder "clip_files".

## Structure

The following source files are required to execute the various experiments of our paper:
- ```baselines.py```: Code which performs training and evaluation of the baseline end-to-end supervised model.
- ```noisy_clip_dataparallel.py```: Performs training and evaluation of the student model, based on the CLIP architecture.
- ```linear_probe.py```: Performs training and evaluation of a linear probe on top of the learned representations.
- ```noise_level_testing.py```: Evaluation of a trained model on various noise levels added in the input.
- ```ood_testing.py```: Evaluation of a trained model on various noise levels added in the input, on changed labels for the dataset.
- ```transfer_learning.py```: Train a linear classifier on a dataset other than ImageNet.
- ```transfer_noiselevels.py```: Evaluate the aforementioned linear classifier.
- ```denoising_baselines.py```: File used for Gaussian denoising.
- ```inpiainting_test.py```: Inpainting of images with missing pixels.
- ```utils.py```: General library for functions used throughout our code.

We also provide ```slice_imagenet100.py```, to be used one time to generate the ImageNet-100 subset we used, as defined by ```imagenet100.txt```. In order to run most of the code we provide, please first run this file with the proper source path to the full ImageNet dataset (can be downloaded separately at https://image-net.org/download) and desired destination path for the 100-class subset. Then, provide the path to your 100-class ImageNet subset in the yaml config files. For further details, refer to the comments in ```slice_imagenet100.py``` and the global variables set at the beginning of the script. The ```slice_imagenet100c.py``` file works in a similar fashion.

In the ```config/``` folder, some sample configuration files for our experiments are included.

## ImageNet with changed labels

For our experiment with changing labels of some ImageNet classes, create a folder ```imagenet100_extra```, containing the below wnids:
- n07930864 (cup)
- n01978287 (Dungeness crab)
- n03792782 (mountain bike)
- n02325366 (wood rabbit)
- n02108915 (French bulldog)

## Examples

Using the following snippets of code, the experiments described in the paper can be run. Note that editing the ```batch_size``` and ```gpus``` parameters of the sample files can lead to speedup and increased performance for the contrastive models. Also note that some of the sample config files, such as those for transfer learning evaluation, require you to provide a path to a saved model checkpoint.

- ```python baselines.py --config_file config/Supervised_CLIP_Baselines/sample.yaml```: Train a baseline model, in an end-to-end supervised fashion.
- ```python noisy_clip_dataparallel.py --config_file config/NoisyRN101/sample.yaml```: Trains a CLIP model using contrastive learning.
- ```python linear_probe.py --config_file config/LinearProbeSubset/sample.yaml```: Trains a linear probe on top of a representation learned using contrastive loss. This requires the user to specify a checkpoint file in the yaml config file.
- ```python noise_level_testing.py --config_file config/NoiseLevelTesting/sample.yaml```: Evaluates a trained model for various levels of noise in the dataset. This requires the user to specify a checkpoint file in the yaml config file. It can also be used to evaluate a model on fixed level of noise. Also see ```sample_denoised.yaml``` for the denoising baseline in the supplementary material.
- ```python ood_testing.py --config_file config/NoiseLevelTesting/sample_ood.yaml```: Similar to the previous one, but this time the dataset required is the one containing the classes with the altered labels.
- ```python transfer_learning.py --config_file config/TransferLearning/sample.yaml```: Train a model on top of learned representations on a different dataset.
- ```python transfer_noiselevels.py --config_file config/Transfer_NoiseLevels/sample.yaml```: Evaluate the above model on various noise levels.
- ```python imagenet100c.py --config_file config/ImageNet100C/sample.yaml```: Evaluate a model on ImageNet100C.
- ```python inpainting_test.py --config_file config/inpainting_sample.yaml```: Evaluate a trained baseline model on masked, then inpainted ImageNet-100 images.

## Saved Models

We provide some pre-trained models for evaluation and finetuning. The models include a contrastively-trained robust encoder backbone for: random masking of 90% pixels, Gaussian blur with kernel size 37 and std 9, and Gaussian additive noise with std 0.5. In addition, we provide linear probes for each of these backbones, trained on distorted ImageNet-100 data. We also provide supervised baselines for each of the above distortions.

To use these models for evaluation, refer to the section above for examples. Specifically, the checkpoints we provide are for fixed distortions on ImageNet-100 data, so ```noise_level_testing.py``` is the most relevant program to evaluate these models. You must alter the given sample config files to point to the path where these checkpoints are contained, as well as to reflect the distortion type for each model. 

Links to checkpoint files:
- Gaussian Blur n=37, std=9 robust backbone: https://drive.google.com/file/d/1q1g0DjY8SCCnkMnt9lN4bLo1352qZFvi/view?usp=sharing
- Random Masking 90% robust backbone: https://drive.google.com/file/d/1HOi7fU-AkOKcVwibDRe97IYnS0U8ijPm/view?usp=sharing
- Gaussian Noise std=0.5 robust backbone: https://drive.google.com/file/d/1PmquZJLCsIHvwTCNuNmfYolWiW1xPzSm/view?usp=sharing

- Gaussian Blur n=37, std=9 linear probe: https://drive.google.com/file/d/1f1llrEZPfcTiYK2E7IO54Cd_sUhIVPFD/view?usp=sharing
- Random Masking 90% linear probe: https://drive.google.com/file/d/1obQCzf74jSN_lOlozTlFVrtaD549teYl/view?usp=sharing
- Gaussian Noise std=0.5 linear probe: https://drive.google.com/file/d/1cl5tu8yq9vpkZGifUyBwMvfFPd9I4Cgd/view?usp=sharing

- Gaussian Blur n=37, std=9 baseline: https://drive.google.com/file/d/1nIfjNjbAhjNsWQgooh_MhmK0qdYb6ThC/view?usp=sharing
- Random Masking 90% baseline: https://drive.google.com/file/d/1jgwu57OQc5qWyzGUVV7tnDyELlxwnAqN/view?usp=sharing
- Gaussian Noise std=0.5 baseline: https://drive.google.com/file/d/1OuLbKoUuLxIxP3-QLdrp9eaE81O2CDag/view?usp=sharing

## References

If you found this repo or our paper useful, please consider citing our work:

```bibtex
@misc{ravula2021inverse,
  title={Inverse Problems Leveraging Pre-trained Contrastive Representations},
  author={Sriram Ravula and Georgios Smyrnis and Matt Jordan and Alexandros G. Dimakis},
  year={2021},
  eprint={2110.07439},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
