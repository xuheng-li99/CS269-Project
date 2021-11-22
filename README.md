# Code to Accompany _Pruning Neural Networks at Initialization: Why Are We Missing the Mark?_

This codebase is a fork of the OpenLTH codebase created by Facebook. For details on basic usage of the codebase, see [https://github.com/facebookresearch/open_lth](https://github.com/facebookresearch/open_lth).

## 1. Creating Networks to Prune

To create a network to prune, use the `train` option for OpenLTH:

```
python open_lth.py train --default_hparams=cifar_resnet_20
```

If you wish to explore pruning at steps other than the beginning and end of training, you can add an additional flag to save the weights at other steps:

```
python open_lth.py train --default_hparams=cifar_resnet_20 --weight_save_steps=1000it,2000it,3000it
```

The above command will save the weights at iterations 1000, 2000, and 3000 for later use.

## 2. Pruning a Network

To prune a network, we use the `branch` functionality of OpenLTH. (We have refactored this functionality slightly to make it possible to create branches of training jobs.)
We have created a branch called `oneshot` that can be found in `training/branch/oneshot_experiments.py`.
This branch makes it possible to prune the network to various sparsities using each of the pruning methods.

For example, the command
```
python open_lth.py branch train oneshot --default_hparams=cifar_resnet_20 --strategy=magnitude --prune_fraction=0.75
```
will prune the ResNet-20 we created in (1) to 75% sparsity using magnitude pruning at initialization. It will then train the network normally from there.

### Strategies and Iterative Pruning

The available values for the `strategy` flag include:
* `random`
* `magnitude`
* `snipN` (uses `N` examples per class to compute the scores)
* `graspN` (uses `N` examples per class to compute the scores)
* `graspabsN` (uses `N` examples per class to compute the scores)
* `synflow`

By default, all of these methods will use one-shot pruning. To make the method iterative, set the `--prune_iterations` flag to the desired number of pruning iterations (e.g., 100 for SynFlow).

By default, this branch will always prune scores with the lowest values. For GraSP, this is the incorrect behavior. To prune the scores with the highest scores (or to invert a pruning method where appropriate), set the `--prune_highest` flag.

### Pruning at a Different Iteration

To prune using the state of the network at a different iteration, set the `--prune_step` and `--state_step` flags to the desired iteration (e.g., `1000it`). You can only use the state of the network if you saved it in (1). Step 0 and the last step of training save by default.

### Lottery Ticket Rewinding

To perform lottery ticket rewinding, set `--prune_step` to the last step of training and set `--state_step` to the desired rewinding iteration.

### Randomly Shuffling

Set the `--randomize_layerwise` flag.

### Randomly Reinitializing

Set the `--reinitialize` flag.

### Changing the Initialization Distribution to N(0, 1)

At both (1) and here, add the flag `--model_init=standard_normal`

## 3. Available Models

* `mnist_lenet_300_100`
* `cifar_resnet_20`
* `cifar_vgg_16`
* `imagenet_resnet_50`
* `tinyimagenet_resnet_18`
* `tinyimagenet_modifiedresnet_18`

## 4. Available Datasets

* `mnist`
* `cifar10`
* `tinyimagenet` (the version we use in the main body; need to download, install according to `datasets/tinyimagenet.py`, and add to `platforms/local.py`)
* `tinyimagenet2` (the version we use for Modified ResNet-18; need to download, install according to `datasets/tinyimagenet.py`, and add to `platforms/local.py`)
* `imagenet` (need to download, install according to `datasets/tinyimagenet.py`, and add to `platforms/local.py`)
