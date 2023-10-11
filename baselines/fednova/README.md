---
title: FedNova (NeurIPS 2020)
url: https://arxiv.org/abs/2007.07481
labels: [normalized averaging, heterogeneous optimization, federated learning]
dataset: [non-iid cifar10 dataset, synthetic dataset]
---

# FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization

****Paper:**** [https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)

****Authors:**** *Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor*

****Abstract:**** *In federated learning, heterogeneity in the clients' local datasets and computation speeds results in large variations in the number of local updates performed by each client in each communication round. Naive weighted aggregation of such models causes objective inconsistency, that is, the global model converges to a stationary point of a mismatched objective function which can be arbitrarily different from the true objective. This paper provides a general framework to analyze the convergence of federated heterogeneous optimization algorithms. It subsumes previously proposed methods such as FedAvg and FedProx and provides the first principled understanding of the solution bias and the convergence slowdown due to objective inconsistency. Using insights from this analysis, we propose FedNova, a normalized averaging method that eliminates objective inconsistency while preserving fast error convergence.*


## About this baseline

****What’s implemented:**** *_The code in this baseline reproduces the results from Table-1 in the paper which corresponds to experiments on Non-IID CIFAR dataset._*

****Datasets:**** *_The dataset in the experiment is a Non-IID CIFAR-10 dataset which is partitioned across 16 clients using a Dirichlet distribution with parameter alpha=0.1._*

****Hardware Setup:**** *_The experiment setting consists of 16 clients. The training is done on a single workstation consisting of 8 CPU cores, 32 GB of RAM and an Nvidia A100 GPU. 
The total GPU memory usage for the experiment is ~ 10 GB(1.1 GB per client + main process). Therefore effectively, 8 clients run in parallel in this setup using the default config. 
The total time for a single experiment in this setup is ~ 50 minutes. (or 30 second per communication round). In case of resource constraints, 
the experiment can be run with 4 clients in parallel by setting client_resources  as following: {num_cpus: 1, num_gpus: 0.25}. This uses ~ 5.4 GB of GPU memory and 4 CPU cores._*

****Contributors:**** *_Aasheesh Singh (Github: [@ashdtu](https://github.com/ashdtu)), MILA-Quebec AI Institute_*


## Experimental Setup

****Task:**** The task is an Image classification task on the  Non-IID CIFAR-10 dataset(10 classes).

****Model:**** We use a standard VGG-11 model for the image classification task. 

****Dataset:**** Each client would have a highly skewed distribution of class labels following the Drichlet distribution.

****Training Hyperparameters:**** Include a table with all the main hyperparameters in your baseline. Please show them with their default value. 


## Environment Setup

``` python
# Navigate to baselines/fednova
cd baselines/fednova

# Set python version
pyenv install 3.10.11
pyenv local 3.10.11

# Tell poetry to use python 3.10
poetry env use 3.10.11

# install the base Poetry environment, make sure there is no existing poetry.lock file and pyproject.toml file is located in the current directory
poetry install

# activate the environment
poetry shell
```

## Running the Experiments
Once the poetry environment is active, you can use the following command to run the various experiments in the Table 1.
You would need to specify the below two command line parameters to iterate through various experiment configurations. They are as follows:

1. `experiment`: This parameter specifies the local optimizer configuration. It can take the following values:
    - `vanilla`: This corresponds to the vanilla SGD as the client optimizer
    - `momentum`: This corresponds to the SGD optimizer with momentum.
    - `proximal`: This corresponds to the SGD optimizer with proximal term in the loss.
    - `hybrd`: This corresponds to hybrid momentum scheme where both the local client optimizer and the server maintain a momentum buffer (only for FedNova strategy).

2. `strategy`: This specifies the aggregation strategy for the client updates. The default is `fednova`. 
If you do not specify this parameter, all experiments will run with FedNova as the strategy and reproduce the rightmost columns of Table-1. 
It can take the following values:
    - `fednova`: This corresponds to the FedNova(default) aggregation strategy. 
    - `fedavg`: This corresponds to the FedAvg aggregation strategy. The left column of Table-1 can be reproduced by setting this parameter.

3. `var_local_epochs`: Takes value True/False. This parameter specifies whether the number of local training epochs for each client are fixed(Epochs=2) or variable(Uniform sampled from [2,5) ). 
It takes the following values:
   - `False`: (default) This corresponds to the fixed number of local epochs for each client. This corresponds to the first part of the Table-1.
   - `True`: This corresponds to the variable number of local epochs for each client. This corresponds to the second part of the Table-1.

```bash  
# general format
python -m fednova.main +experiment=<exp_value> strategy=<strategy_value (fednova/fedavg)> var_local_epochs=<var_local_epochs_value (True/False)>

# example script
python -m fednova.main +experiment=momentum strategy=fednova var_local_epochs=True

```


## Expected Results

Centralized Evaluation: Accuracy(in %) on centralized Test set

| Local Epochs | Client Optimizer | FedAvg | FedNova |
| ------------ | ---------------- | ------ | ------- |
| 2            | Vanilla          | 71.57  | 68.25   |
| 2            | Momentum         | 75.92  | 73.33   |
| 2            | Proximal         | -  | 67.47   |
| Random(2-5)  | Vanilla          | -  | 72.27   |
| Random(2-5)  | Momentum         | -  | 75.73   |
| Random(2-5)  | Proximal         | -  | 71.48   |
| Random(2-5)  | Server           |  N/A   |  -      |
| Random(2-5)  | Hybrid           |   N/A    |  -      |
