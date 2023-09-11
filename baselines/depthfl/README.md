---
title: DepthFL:Depthwise Federated Learning for Heterogeneous Clients
url: https://openreview.net/forum?id=pf8RIZTMU58
labels: [image classification, system heterogeneity] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [CIFAR100] # list of datasets you include in your baseline
---

# DepthFL: Depthwise Federated Learning for Heterogeneous Clients

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** : https://openreview.net/forum?id=pf8RIZTMU58

****Authors:**** : Minjae Kim, Sangyoon Yu, Suhyun Kim, Soo-Mook Moon

****Abstract:**** : Federated learning is for training a global model without collecting private local data from clients. As they repeatedly need to upload locally-updated weights or gradients instead, clients require both computation and communication resources enough to participate in learning, but in reality their resources are heterogeneous. To enable resource-constrained clients to train smaller local models, width scaling techniques have been used, which reduces the channels of a global model. Unfortunately, width scaling suffers from heterogeneity of local models when averaging them, leading to a lower accuracy than when simply excluding resource-constrained clients from training. This paper proposes a new approach based on depth scaling called DepthFL. DepthFL defines local models of different depths by pruning the deepest layers off the global model, and allocates them to clients depending on their available resources. Since many clients do not have enough resources to train deep local models, this would make deep layers partially-trained with insufficient data, unlike shallow layers that are fully trained. DepthFL alleviates this problem by mutual self-distillation of knowledge among the classifiers of various depths within a local model. Our experiments show that depth-scaled local models build a global model better than width-scaled ones, and that self-distillation is highly effective in training data-insufficient deep layers.


## About this baseline

****What’s implemented:**** The code in this directory replicates the experiments in DepthFL: Depthwise Federated Learning for Heterogeneous Clients (Kim et al., 2023) for CIFAR100, which proposed the DepthFL algorithm. Concretely, it replicates the results for CIFAR100 dataset in Table 2,3 and 4.

****Datasets:**** CIFAR100 from PyTorch's Torchvision

****Hardware Setup:**** These experiments were run on a server with Nvidia 3090 GPUs. Any machine with 1x 8GB GPU or more would be able to run it in a reasonable amount of time. 

****Contributors:**** Minjae Kim


## Experimental Setup

****Task:**** Image Classification

****Model:**** ResNet18

**Dataset:** This baseline only includes the CIFAR100 dataset. By default it will be partitioned into 100 clients following IID distribution. The settings are as follow:

| Dataset | #classes | #partitions | partitioning method |
| :------ | :---: | :---: | :---: |
| CIFAR100 | 100 | 100 | IID |

**Training Hyperparameters:**
The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python main.py` directly)

| Description | Default Value |
| ----------- | ----- |
| total clients | 100 |
| local epoch | 5 |
| batch size | 50 |
| number of rounds | 1000 |
| participation ratio | 10% |
| learning rate | 0.1 |
| learning rate decay | 0.998 |
| client resources | {'num_cpus': 1.0, 'num_gpus': 0.5 }|
| data partition | IID |
| optimizer | SGD with dynamic regularization |
| alpha | 0.1 |


## Environment Setup

To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```


## Running the Experiments

To run this DepthFL, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
python -m depthfl.main # this will run using the default settings in the `conf/config.yaml`

# you can override settings directly from the command line
python -m depthfl.main exclusive_learning=true model_size=1 # exclusive learning - 100% (a)
python -m depthfl.main exclusive_learning=true model_size=4 # exclusive learning - 25% (d)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false # DepthFL (FedAvg)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false fit_config.extended=false # InclusiveFL
```

To run using HeteroFL:
```bash
# since sbn takes too long, we test global model every 50 rounds. 
python -m fedprox.main --config-name="heterofl" # HeteroFL
python -m fedprox.main --config-name="heterofl" exclusive_learning=true model_size=1 # exclusive learning - 100% (a)
```

### Stateful clients comment

To implement feddyn, stateful clients that store prev_grads information are needed. Since flwr does not yet officially support stateful clients, it was implemented as a temporary measure by loading prev_grads from disk when creating a client, and then storing it again on disk after learning. Specifically, there are files that store the state of each client in the prev_grads folder.


## Expected Results

With the following command we run DepthFL (FedDyn / FedAvg), InclusiveFL, and HeteroFL to replicate the results of table 2,3,4 in DepthFL paper. Tables 2, 3, and 4 may contain results from the same experiment in multiple tables. 

```bash
# table 2
python -m depthfl.main # table 2 & 4
python -m depthfl.main exclusive_learning=true model_size=1
python -m depthfl.main exclusive_learning=true model_size=2
python -m depthfl.main exclusive_learning=true model_size=3
python -m depthfl.main exclusive_learning=true model_size=4

# table 2 & 3
python -m depthfl.main --config-name="heterofl" 
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=1 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=2 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=3 model.scale=false
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=4 model.scale=false
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=1
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=2
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=3
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=4

# table 3
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false fit_config.extended=false

# table 4
python -m depthfl.main fit_config.kd=false
```

The above commands would generate results in DepthFL paper. The numbers below are the results of a single run, and although they do not perfectly match the numbers recorded in the paper, they are very close.

**Table 2** 

100% (a), 75%(b), 50%(c), 25% (d) cases are exclusive learning scenario. 100% (a) exclusive learning means, the global model and every local model are equal to the smallest local model, and 100% clients participate in learning. Likewise, 25% (d) exclusive learning means, the global model and every local model are equal to the larget local model, and only 25% clients participate in learning.

| Scaling Method | Dataset | Global Model | 100% (a) | 75% (b) | 50% (c) | 25% (d) | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| HeterFL | CIFAR100 | 57.61 | 64.39 | 66.08 | 62.03 | 51.99 |
| DepthFL (FedAvg) | CIFAR100 | 72.67 | 67.08 | 70.78 | 68.41 | 59.17 |
| DepthFL | CIFAR100 | 76.06 | 69.68 | 73.21 | 70.29 | 60.32 |

**Table 3** 

Accuracy of global sub-models compared to exclusive learning on CIFAR-100.

| Method | Algorithm | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Width Scaling | Exclusive Learning | 64.39 | 66.08 | 62.03 | 51.99 |
| Width Scaling | HeteroFL | 51.08 | 55.89 | 58.29 | 57.61 |

| Method | Algorithm | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Depth Scaling | Exclusive Learning | 67.08 | 68.00 | 66.19 | 56.78 |
| Depth Scaling | InclusiveFL | 47.61 | 53.88 | 59.48 | 60.46 |
| Depth Scaling | DepthFL (FedAvg)| 66.18 | 67.56 | 67.97 | 68.01 |

**Table 4** 

Accuracy of the global model with/without self distillation on CIFAR-100.

| Distribution | Dataset | KD | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 | Ensemble | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IID | CIFAR-100 | &cross; | 70.13 | 69.63 | 68.92 | 68.92 | 74.48 | 
| IID | CIFAR-100 | &check; | 71.74 | 73.35 | 73.57 | 73.55 | 76.06 | 


