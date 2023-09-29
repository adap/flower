---
title: FedMix Approximation of Mixup under Mean Augmented Federated Learning
url: https://arxiv.org/abs/2107.00233
labels: ["data heterogeneity", "mixup", "non-iid"]
dataset: ["cifar10", "femnist"]
---

# FedMix: Approximation of Mixup under Mean Augmented Federated Learning

****Paper:**** https://arxiv.org/abs/2107.00233

****Authors:**** Tehrim Yoon, Sumin Shin, Sung Ju Hwang, Eunho Yang

****Abstract:**** Federated learning (FL) allows edge devices to collectively learn a model without directly sharing data within each device, thus preserving privacy and eliminating the need to store data globally. While there are promising results under the assumption of independent and identically distributed (iid) local data, current state-of-the-art algorithms suffer from performance degradation as the heterogeneity of local data across clients increases. To resolve this issue, we propose a simple framework, Mean Augmented Federated Learning (MAFL), where clients send and receive averaged local data, subject to the privacy requirements of target applications. Under our framework, we propose a new augmentation algorithm, named FedMix, which is inspired by a phenomenal yet simple data augmentation method, Mixup, but does not require local raw data to be directly shared among devices. Our method shows greatly improved performance in the standard benchmark datasets of FL, under highly non-iid federated settings, compared to conventional algorithms.


## About this baseline

****What’s implemented:**** The code in this directory implements two of the Federated Strategies mentioned in the paper: NaiveMix and FedMix

****Datasets:**** CIFAR10, FEMNIST

****Hardware Setup:**** Experiments in this baseline were run on 2x Nvidia Tesla V100 16GB.

****Contributors:**** [DevPranjal](https://github.com/DevPranjal)


## Experimental Setup

****Task:**** Image Classification

****Model:**** Models used are modified versions of existing known models and are descirbed in Appendix B (Experimental Details). For the CIFAR10 dataset, the authors use a modified version of VGG, while LeNet-5 is used for the FEMNIST dataset.

****Dataset:**** Both the datasets used (CIFAR10 and FEMNIST) incorporate data heterogenity in diferent ways to simulate a non-iid setting. For the CIFAR10 experiment, data is allocated such that each client has data from only a selected number of randomly chosen classes. For the FEMNIST experiment, data is allocated such that each client has data from only one writer, resulting in 200-300 samples per client on average.

| Property | CIFAR10 Partioning | FEMNIST Partitioning |
| -- | -- | -- |
| num classes | 10 | 62 |
| num clients | 60 | 100 |
| num classes per client | 2 | |
| non-iidness type | data from selected classes | data from single writer |


****Training Hyperparameters:****

| Hyperparameter | CIFAR10 Experiments | FEMNIST Experiments |
| -- | -- | -- |
| local epochs | 2 | 10 |
| local learning rate | 0.01 | 0.01|
| local lr decay after round | 0.999 | 0.999 |
| local batch size | 10 | 10 |
| num classes per client | 2 | |
| total number of clients | 60 | 100 |
| clients per round | 15 | 10 |
| mash batch size | full local data (`all`) | full local data (`all`)|
| mixup ratio | 0.05 | 0.2 |
| num rounds | 500 | 200 |


## Environment Setup

```
# set local python version via pyenv
pyenv local 3.10.6

# then fix that for poetry
poetry env use 3.10.6

# then install poetry env
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

To run an experiment with default hyperparameters (as mentioned in the paper), execute:

```
python -m fedmix.main +experiment={dataset}_{strategy}
# dataset can be: cifar10, femnist
# strategy can be: fedavg, naivemix, fedmix
```

To run custom experiments: create a new `.yaml` file in `conf/experiment` and override the default hyperparameters.

## Expected Results

