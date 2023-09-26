---
title: FedExP Speeding Up Federated Averaging via Extrapolation
url: https://openreview.net/forum?id=IPrzNbddXV
labels: [image classification, Optimization,  Step Size]
dataset: [CIFAR-10, CIFAR-100]
---

# FedExP : Speeding Up Federated Averaging via Exptrapolation

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** [openreview.net/forum?id=IPrzNbddXV](https://openreview.net/forum?id=IPrzNbddXV)

****Authors:**** : Divyansh Jhunjhunwala, Shiqiang Wang, Gauri Joshi

****Abstract:**** : Federated Averaging (FedAvg) remains the most popular algorithm for Federated Learning (FL) optimization due to its simple implementation, stateless nature, and privacy guarantees combined with secure aggregation. Recent work has sought to generalize the vanilla averaging in FedAvg to a generalized gradient descent step by treating client updates as pseudo-gradients and using a server step size. While
the use of a server step size has been shown to provide performance improvement theoretically, the practical benefit of the server step size has not been seen in most existing works. In this work, we present FedExP, a method to adaptively determine the server step size in FL based on dynamically varying pseudo-gradients throughout the FL process. We begin by considering the overparameterized convex regime, where we reveal an interesting similarity between FedAvg and the Projection Onto Convex Sets (POCS) algorithm. We then show how FedExP can be motivated as a novel extension to the extrapolation mechanism that is used to speed up POCS. Our theoretical analysis later also discusses the implications of FedExP in underparameterized and non-convex settings. Experimental results show that
FedExP consistently converges faster than FedAvg and competing baselines on a range of realistic FL datasets.


## About this baseline

****What’s implemented:**** : The code in this directory replicates the experiments in the paper(FedExP : Speeding Up Federated Averaging via Exptrapolation), which proposed the FedExP strategy. Specifically, it replicates the results for For Cifar10 and Cifar100 in Figure 3.

****Datasets:**** : Cifar10 and Cifar100 from PyTorch's Torchvision

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** : Omar Mokhtar and Roeia Amr


## Experimental Setup

****Task:**** : Image classification

****Model:**** : This directory implements the ResNet-18 Model:
The ResNet-18 model is employed in the paper as the core architecture for experiments on CIFAR-10 and CIFAR-100 datasets.

****Dataset:**** :
The baseline utilizes both CIFAR-10 and CIFAR-100 datasets, which will be distributed among 100 clients. The Dirichlet distribution is employed to introduce variability in the composition of client datasets for CIFAR.

| Dataset  | #classes | #partitions |  partitioning method   |
|:---------|:--------:|:-----------:|:----------------------:|
| Cifar10  |    10    |     100     | Dirichlet distribution |
| Cifar100 |   100    |     100     | Dirichlet distribution |


****Training Hyperparameters:**** :
The following tables shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python main.py` directly)

| Description                 | Default Value                       |
|-----------------------------|-------------------------------------|
| total clients               | 100                                 |
| clients per round           | 20                                  |
| number of rounds            | 500                                 |
| number of local rounds      | 20                                  |
| batch_size                  | 50                                  |
| client resources            | {'num_cpus': 2.0, 'num_gpus': 0.2 } |
| eta_l (local learning rate) | 0.01                                |
| epsilon                     | 0.001                               |
| decay                       | 0.998                               |
| weight_decay                | 0.0001                              |
| max_norm                    | 10                                  |

For Dataset:
Choice of alpha parameter for the Dirichlet distribution used to create heterogeneity in the client datasets for CIFAR

| Description | Default Value |
|-------------|---------------|
| alpha       | 0.5           |

## Environment Setup

To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell

#install required libs
# install PyTorch with GPU support. Please note this baseline is very lightweight so it can run fine on a CPU.

```

## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m <baseline-name>.main <no additional arguments> # where <baseline-name> is the name of this directory and that of the only sub-directory in this directory (i.e. where all your source code is)

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

poetry run -m <baseline-name>.dataset_preparation <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

poetry run -m <baseline-name>.main  <override_some_hyperparameters>
.
.
.
poetry run -m <baseline-name>.main  <override_some_hyperparameters>
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
