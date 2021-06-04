## Requirements

In order to run the code for our models, it is necessary to install ```pytorch_lightning``` and all of its dependencies. The file ```requirements.txt``` provides the environment specifications for running our coder. Moreover, it is necessary that the following files from the OpenAI CLIP repository (https://github.com/openai/CLIP) are added, along with their respective requirements:
- ```model.py```
- ```clip.py```
- ```simple_tokenizer.py```
- ```bpe_simple_vocab_16e6.txt.gz```

For simplicity, we have copied these required files into our own code.

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

We also provide ```slice_imagenet100.py```, a code to be used one time to generate the ImageNet-100 subset we used, as defined by ```imagenet100.txt```. In order to run most of the code we provide, please first run this file with the proper source path to the full ImageNet dataset (can be downloaded separately at https://image-net.org/download) and desired destination path for the 100-class subset. Then, provide the path to your 100-class ImageNet subset in the yaml config files. For further details, refer to the comments in ```slice_imagenet100.py``` and the global variables set at the beginning of the script. The ```slice_imagenet100c.py``` file works in a similar fashion.

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
