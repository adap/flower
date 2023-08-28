---
title: Measuring the effects of non-identical data distribution for federated visual classification
url: https://arxiv.org/abs/1909.06335
labels: [non-iid, image classification] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [CIFAR-10, FMNIST] # list of datasets you include in your baseline
---

# FedAvgM: Measuring the effects of non-identical data distribution for federated visual classification

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://arxiv.org/abs/1909.06335

**Authors:** Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown

**Abstract:** Federated Learning enables visual models to be trained in a privacy-preserving way using real-world data from mobile devices. Given their distributed nature, the statistics of the data across these devices is likely to differ significantly. In this work, we look at the effect such non-identical data distributions has on visual classification via Federated Learning. We propose a way to synthesize datasets with a continuous range of identicalness and provide performance measures for the Federated Averaging algorithm. We show that performance degrades as distributions differ more, and propose a mitigation strategy via server momentum. Experiments on CIFAR-10 demonstrate improved classification performance over a range of non-identicalness, with classification accuracy improved from 30.1% to 76.9% in the most skewed settings.


## About this baseline

**What’s implemented:** The code in this directory evaluates the effects of non-identical data distribution for visual classification task based on paper _Measuring the effects of non-identical data distribution for federated visual classification_ (Hsu et al., 2019). It reproduces the FedAvgM and FedAvg performance curves for different non-identical-ness of the dataset (CIFAR-10 and FEMNIST). _Figure 5 in the paper, section 4.2._

**Datasets:** CIFAR-10, and FMNIST

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** Gustavo de Carvalho Bertoli

## Experimental Setup

**Task:** Image Classification

**Model:** This directory implements the same CNN model presented in the following paper (`models.py`):

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics (pp. 1273-1282). PMLR. ([Link](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)):

As the following excerpt:

"*A CNN with two 5x5 convolution layers (the first with 32 channels, the second with 64, each followed with 2x2 max pooling), a fully connected layer with 512 units and ReLu activation, and a final softmax output layer (1,663,370 total parameters)"*

:warning: However, this architecture implemented on this baseline results in 878,538 parameters. Regarding this architecture, the historical references mentioned on the FedAvgM paper are [this](https://web.archive.org/web/20190415103404/https://www.tensorflow.org/tutorials/images/deep_cnn) and [this](https://web.archive.org/web/20170807002954/https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py).

**Dataset:** This baseline includes the CIFAR-10 and FMNIST datasets. By default it will run with the CIFAR-10. The data partition uses a configurable Latent Dirichlet Allocation (LDA) distribution (`concentration` parameter equals 0.1 as default) to create **non-iid distributions** between the clients. The understanding for this `concentration` (α) is that α→∞ all clients have identical distribution, and α→𝟢 each client hold samples from only one class.

| Dataset | # classes | # partitions | partition method | partition settings|
| :------ | :---: | :---: | :---: | :---: |
| CIFAR-10 | 10 | `num_clients` | Latent Dirichlet Allocation (LDA) | `concentration` |
| FMNIST | 10 | `num_clients` | Latent Dirichlet Allocation (LDA) | `concentration` |

**Training Hyperparameters:**
The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python main.py` directly)

| Description | Default Value |
| ----------- | ----- |
| total clients | 10 |
| number of rounds | 5 |
| model | CNN |
| strategy | FedAvgM |
| dataset | CIFAR-10 |
| concentration | 0.1 |
| server momentum | 0.9 |
| server learning rate | 1.0 |
| server reporting fraction | 0.05 |
| client local epochs | 1 |
| client batch size | 64 |

## Specifying the Python Version
This baseline was tested with Python 3.9.16.

By default, Poetry will use the Python version in your system. In some settings, you might want to specify a particular version of Python to use inside your Poetry environment. You can do so with [`pyenv`](https://github.com/pyenv/pyenv). Check the documentation for the different ways of installing `pyenv`, but one easy way is using the [automatic installer](https://github.com/pyenv/pyenv-installer):

```bash
curl https://pyenv.run | bash # then, don't forget links to your .bashrc/.zshrc
```

You can then install any Python version with `pyenv install <python-version>` (e.g. `pyenv install 3.9.16`). Then, in order to use that version for this baseline, you'd do the following:

```bash
# cd to your baseline directory (i.e. where the `pyproject.toml` is)
pyenv local 3.9.16

# set that version for poetry
poetry env use 3.9.16

# then you can install your Poetry environment (see the next setp)
```

## Environment Setup

This baseline works with TensorFlow 2.10, no additional step required once using Poetry to set up the environment.

We use Poetry to manage the Python environment for each individual baseline. You can follow the instructions [here](https://python-poetry.org/docs/) to install Poetry in your machine. 

To construct the Python environment with Poetry follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

To run this FedAvgM with CIFAR-10 baseline, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash  
poetry run python -m fedavgm.main # this will run using the default setting in the `conf/base.yaml`

# you can override settings directly from the command line

poetry run python -m fedavgm.main strategy=fedavg num_clients=1000 num_rounds=50 # will set the FedAvg with 1000 clients and 50 rounds

poetry run python -m fedavgm.main dataset=fmnist noniid.concentration=10 # use the FMNIST dataset and a different concentration for the LDA-based partition

poetry run python -m fedavgm.main server.reporting_fraction=0.2 client.local_epochs=5 # will set the reporting fraction to 20% and the local epochs in the clients to 5
```

## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m fedavgm.main --multirun client.local_epochs=1,5 noniid.concentration=100,10,1,0.5,0.2,0.1,0.05,0 strategy=fedavgm,fedavg server.reporting_fraction=0.05,0.1,0.4 num_rounds=10000 num_clients=100

# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

poetry run python -m fedavgm.main --multirun client.local_epochs=1,5 noniid.concentration=100,10,1,0.5,0.2,0.1,0.05,0 strategy=fedavgm,fedavg server.reporting_fraction=0.05,0.1,0.4 num_rounds=10000 dataset=fmnist num_clients=100

# add more commands + plots for additional experiments.
```
