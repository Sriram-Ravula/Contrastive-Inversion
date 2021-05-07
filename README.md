# Creating Robust Representations from Pre-Trained Image Encoders using Contrastive Learning
## Sriram Ravula, Georgios Smyrnis

This is the code for our project "Creating Robust Representations from Pre-Trained Image Encoders using Contrastive Learning".  We make use of contrastive learning and OpenAI's CLIP to find good embeddings for images with lossy transformations.

## Requirements

In order to run the code for our models, it is necessary to install ```pytorch_lightning``` and all of its dependencies. Moreover, it is necessary that the following files from the OpenAI CLIP repository (https://github.com/openai/CLIP) are added, along with their respective requirements:
- ```model.py```
- ```clip.py```
- ```simple_tokenizer.py```
- ```bpe_simple_vocab_16e6.txt.gz```
For simplicity, we have copied these required files into our own code. To run the examples, make sure to download some checkpoint files, available here: https://drive.google.com/drive/folders/1wvm8lMSsQZU8YF_AXIhdqsHV7x6gg9rZ?usp=sharing.

## Structure

The following source files are required to execute the various experiments mentioned in our report:
- ```baselines.py```: Code which performs training and evaluation of the baseline end-to-end supervised model.
- ```noisy_clip_dataparallel.py```: Performs training and evaluation of the student model, based on the CLIP architecture.
- ```zeroshot_validation.py```: Performs evaluation of the zero-shot model.
- ```linear_probe.py```: Performs training and evaluation of a linear probe on top of the learned representations.
- ```noise_level_testing.py```: Evaluation of a trained model on various noise levels added in the input.
- ```utils.py```: General library for functions used throughout our code.

We also provide ```slice_imagenet100.py```, a code to be used one time to generate the ImageNet-100 subset we used, as defined by ```imagenet100.txt```. In order to run most of the code we provide, please first run this file with the proper source path to the full ImageNet dataset (can be downloaded separately at https://image-net.org/download) and desired destination path for the 100-class subset. Then, provide the path to your 100-class ImageNet subset in the yaml config files. For further details, refer to the comments in ```slice_imagenet100.py``` and the global variables set at the beginning of the script.

In the ```config/``` folder, some sample configuration files for our experiments are included.

## Examples

Using the following snippets of code, the experiments described in the report can be run. Note that editing the ```batch_size``` and ```gpus``` parameters of the sample files will lead to speedup and increased performance for the contrastive models. 

- ```python baselines.py --config_file config/Supervised_CLIP_Baselines/sample.yaml```: Train a baseline model, in an end-to-end supervised fashion.
- ```python noisy_clip_dataparallel.py --config_file config/NoisyRN101/sample.yaml```: Trains a CLIP model using contrastive learning.
- ```python zeroshot_validation.py --config_file config/NoisyRN101/sample.yaml --ckpt_file rand90_zeroshot.ckpt```: Performs zeroshot evaluation of a trained zero-shot clip model. The sample file to be used is the same one specified during training (for flexibility, checkpoint file provided separately).
- ```python linear_probe.py --config_file config/LinearProbeSubset/sample.yaml```: Trains a linear probe on top of a representation learned using contrastive loss. This requires the user to specify a checkpoint file in the yaml config file.
- ```python noise_level_testing.py --config_file config/NoiseLevelTesting/sample.yaml```: Evaluates a trained model for various levels of noise in the dataset. This requires the user to specify a checkpoint file in the yaml config file.
