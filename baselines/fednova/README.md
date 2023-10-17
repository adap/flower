---
title: FedNova (NeurIPS 2020)
url: https://arxiv.org/abs/2007.07481
labels: [normalized averaging, heterogeneous optimization, federated learning]
dataset: [non-iid cifar10 dataset]
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

****Task:**** The task is Image classification on CIFAR-10 dataset.

****Model:**** The experiment uses the below configuration of VGG-11 model for the image classification task. 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
              ReLU-2           [-1, 64, 32, 32]               0
         MaxPool2d-3           [-1, 64, 16, 16]               0
            Conv2d-4          [-1, 128, 16, 16]          73,856
              ReLU-5          [-1, 128, 16, 16]               0
         MaxPool2d-6            [-1, 128, 8, 8]               0
            Conv2d-7            [-1, 256, 8, 8]         295,168
              ReLU-8            [-1, 256, 8, 8]               0
            Conv2d-9            [-1, 256, 8, 8]         590,080
             ReLU-10            [-1, 256, 8, 8]               0
        MaxPool2d-11            [-1, 256, 4, 4]               0
           Conv2d-12            [-1, 512, 4, 4]       1,180,160
             ReLU-13            [-1, 512, 4, 4]               0
           Conv2d-14            [-1, 512, 4, 4]       2,359,808
             ReLU-15            [-1, 512, 4, 4]               0
        MaxPool2d-16            [-1, 512, 2, 2]               0
           Conv2d-17            [-1, 512, 2, 2]       2,359,808
             ReLU-18            [-1, 512, 2, 2]               0
           Conv2d-19            [-1, 512, 2, 2]       2,359,808
             ReLU-20            [-1, 512, 2, 2]               0
        MaxPool2d-21            [-1, 512, 1, 1]               0
          Dropout-22                  [-1, 512]               0
           Linear-23                  [-1, 512]         262,656
             ReLU-24                  [-1, 512]               0
          Dropout-25                  [-1, 512]               0
           Linear-26                  [-1, 512]         262,656
             ReLU-27                  [-1, 512]               0
           Linear-28                   [-1, 10]           5,130
================================================================
Total params: 9,750,922
Trainable params: 9,750,922
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.57
Params size (MB): 37.20
Estimated Total Size (MB): 39.78
----------------------------------------------------------------
```

****Dataset:**** The dataset is Non-IID CIFAR-10 dataset which is partitioned across 16 clients using a Dirichlet distribution Dir16(0.1), (alpha=0.1) as done
in [paper](https://arxiv.org/abs/2002.06440) . Each client gets a different skewed distribution of the class labels following this split.

****Training Hyperparameters:****

| Hyperparameter                  | Description                                                                               | Default value |
|----------------------------------|-------------------------------------------------------------------------------------------|---------------|
| optimizer.learning_rate          | Learning rate of local client optimizers                                                  | 0.05          |
| optimizer.momentum               | Momentum factor                                                                          | 0             |
| optimizer.mu                     | Proximal updates factor                                                                  | 0             |
| optimizer.weight_decay          | Weight decay for regularization                                                          | 1e-4          |
| num_epochs                       | Number of local training epochs for clients                                               | 2             |
| num_rounds                       | Number of server communication rounds                                                    | 100           |
| var_local_epochs                 | Whether to have variable or fixed local client training epochs. If True, samples num_epochs uniformly in (2,5) | False         |
| batch size                       | Batch size for training                                                                   | 32            |




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
| 2            | Proximal         | 72.69  | 67.47   |
| Random(2-5)  | Vanilla          | -  | 72.27   |
| Random(2-5)  | Momentum         | -  | 75.73   |
| Random(2-5)  | Proximal         | -  | 71.48   |
| Random(2-5)  | Server           |  N/A   |  -      |
| Random(2-5)  | Hybrid           |   N/A    |  -      |
