# Creating Robust Representations from Pre-Trained Image Encoders using Contrastive Learning
## Sriram Ravula, Georgios Smyrnis

This is the code for our project "Creating Robust Representations from Pre-Trained Image Encoders using Contrastive Learning".  We make use of contrastive learning and OpenAI's CLIP to find good embeddings for images with lossy transformations. 

## Requirements

In order to run the code for our models, it is necessary to install ```pytorch_lightning``` and all of its dependencies. Moreover, it is necessary that the following files from the OpenAI CLIP repository (https://github.com/openai/CLIP) are added, along with their respective requirements:
- ```model.py```
- ```clip.py```
- ```bpe_simple_vocab_16e6.txt.gz```
For simplicity, we have copied these required files into our own code.

## Structure

The following source files are required to execute the various experiments mentioned in our report:
- ```baselines.py```: Code which performs training and evaluation of the baseline end-to-end supervised model.
- ```noisy_clip_dataparallel.py```: Performs training and evaluation of the student model, based on the CLIP architecture.
- ```zeroshot_validation.py```: Performs evaluation of the zero-shot model.
- ```linear_probe.py```: Performs training and evaluation of a linear probe on top of the learned representations.
- ```noise_level_testing.py```: Evaluation of a trained model on various noise levels added in the input.
- ```utils.py```: General library for functions used throughout our code.

We also provide ```slice_imagenet100.py```, a code to be used one time to generate the ImageNet-100 subset we used, as defined by ```imagenet100.txt```.

## Examples
