---
title: DepthFL:Depthwise Federated Learning for Heterogeneous Clients
url: https://openreview.net/forum?id=pf8RIZTMU58
labels: [image classification, system heterogeneity, cross-device, knowledge distillation]
dataset: [CIFAR-100]
---

# DepthFL: Depthwise Federated Learning for Heterogeneous Clients

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [openreview.net/forum?id=pf8RIZTMU58](https://openreview.net/forum?id=pf8RIZTMU58)

**Authors:** Minjae Kim, Sangyoon Yu, Suhyun Kim, Soo-Mook Moon

**Abstract:** Federated learning is for training a global model without collecting private local data from clients. As they repeatedly need to upload locally-updated weights or gradients instead, clients require both computation and communication resources enough to participate in learning, but in reality their resources are heterogeneous. To enable resource-constrained clients to train smaller local models, width scaling techniques have been used, which reduces the channels of a global model. Unfortunately, width scaling suffers from heterogeneity of local models when averaging them, leading to a lower accuracy than when simply excluding resource-constrained clients from training. This paper proposes a new approach based on depth scaling called DepthFL. DepthFL defines local models of different depths by pruning the deepest layers off the global model, and allocates them to clients depending on their available resources. Since many clients do not have enough resources to train deep local models, this would make deep layers partially-trained with insufficient data, unlike shallow layers that are fully trained. DepthFL alleviates this problem by mutual self-distillation of knowledge among the classifiers of various depths within a local model. Our experiments show that depth-scaled local models build a global model better than width-scaled ones, and that self-distillation is highly effective in training data-insufficient deep layers.


## About this baseline

**What’s implemented:** The code in this directory replicates the experiments in DepthFL: Depthwise Federated Learning for Heterogeneous Clients (Kim et al., 2023) for CIFAR100, which proposed the DepthFL algorithm. Concretely, it replicates the results for CIFAR100 dataset in Table 2, 3 and 4.

**Datasets:** CIFAR100 from PyTorch's Torchvision

**Hardware Setup:** These experiments were run on a server with Nvidia 3090 GPUs. Any machine with 1x 8GB GPU or more would be able to run it in a reasonable amount of time. With the default settings, clients make use of 1.3GB of VRAM. Lower `num_gpus` in `client_resources` to train more clients in parallel on your GPU(s). 

**Contributors:** Minjae Kim


## Experimental Setup

**Task:** Image Classification

**Model:** ResNet18

**Dataset:** This baseline only includes the CIFAR100 dataset. By default it will be partitioned into 100 clients following IID distribution. The settings are as follow:

| Dataset | #classes | #partitions | partitioning method |
| :------ | :---: | :---: | :---: |
| CIFAR100 | 100 | 100 | IID or Non-IID |

**Training Hyperparameters:**
The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python -m depthfl.main` directly)

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
# Set python version
pyenv install 3.10.6
pyenv local 3.10.6

# Tell poetry to use python 3.10
poetry env use 3.10.6

# Install the base Poetry environment
poetry install

# Activate the environment
poetry shell
```


## Running the Experiments

To run this DepthFL, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
# this will run using the default settings in the `conf/config.yaml`
python -m depthfl.main  # 'accuracy' : accuracy of the ensemble model, 'accuracy_single' : accuracy of each classifier.

# you can override settings directly from the command line
python -m depthfl.main exclusive_learning=true model_size=1 # exclusive learning - 100% (a)
python -m depthfl.main exclusive_learning=true model_size=4 # exclusive learning - 25% (d)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false # DepthFL (FedAvg)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false fit_config.extended=false # InclusiveFL
```

To run using HeteroFL:
```bash
# since sbn takes too long, we test global model every 50 rounds. 
python -m depthfl.main --config-name="heterofl" # HeteroFL
python -m depthfl.main --config-name="heterofl" exclusive_learning=true model_size=1 # exclusive learning - 100% (a)
```

### Stateful clients comment

To implement `feddyn`, stateful clients that store prev_grads information are needed. Since flwr does not yet officially support stateful clients, it was implemented as a temporary measure by loading `prev_grads` from disk when creating a client, and then storing it again on disk after learning. Specifically, there are files that store the state of each client in the `prev_grads` folder. When the strategy is instantiated (for both `FedDyn` and `HeteroFL`) the content of `prev_grads` is reset. 


## Expected Results

With the following command we run DepthFL (FedDyn / FedAvg), InclusiveFL, and HeteroFL to replicate the results of table 2,3,4 in DepthFL paper. Tables 2, 3, and 4 may contain results from the same experiment in multiple tables. 

```bash
# table 2 (HeteroFL row)
python -m depthfl.main --config-name="heterofl" 
python -m depthfl.main --config-name="heterofl" --multirun exclusive_learning=true model.scale=false model_size=1,2,3,4 

# table 2 (DepthFL(FedAvg) row)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false 
python -m depthfl.main --multirun fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=1,2,3,4

# table 2 (DepthFL row)
python -m depthfl.main
python -m depthfl.main --multirun exclusive_learning=true model_size=1,2,3,4
```

**Table 2** 

100% (a), 75%(b), 50%(c), 25% (d) cases are exclusive learning scenario. 100% (a) exclusive learning means, the global model and every local model are equal to the smallest local model, and 100% clients participate in learning. Likewise, 25% (d) exclusive learning means, the global model and every local model are equal to the largest local model, and only 25% clients participate in learning.

| Scaling Method | Dataset | Global Model | 100% (a) | 75% (b) | 50% (c) | 25% (d) | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| HeteroFL<br>DepthFL (FedAvg)<br>DepthFL | CIFAR100 | 57.61<br>72.67<br>76.06 | 64.39<br>67.08<br>69.68 | 66.08<br>70.78<br>73.21 | 62.03<br>68.41<br>70.29 | 51.99<br>59.17<br>60.32 |

```bash
# table 3 (Width Scaling - Duplicate results from table 2)
python -m depthfl.main --config-name="heterofl" 
python -m depthfl.main --config-name="heterofl" --multirun exclusive_learning=true model.scale=false model_size=1,2,3,4 

# table 3 (Depth Scaling : Exclusive Learning, DepthFL(FedAvg) rows - Duplicate results from table 2)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false 
python -m depthfl.main --multirun fit_config.feddyn=false fit_config.kd=false  exclusive_learning=true model_size=1,2,3,4

## table 3 (Depth Scaling - InclusiveFL row)
python -m depthfl.main fit_config.feddyn=false fit_config.kd=false fit_config.extended=false
```

**Table 3** 

Accuracy of global sub-models compared to exclusive learning on CIFAR-100.

| Method | Algorithm | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Width Scaling | Exclusive Learning<br>HeteroFL| 64.39<br>51.08 | 66.08<br>55.89 | 62.03<br>58.29 | 51.99<br>57.61 |

| Method | Algorithm | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Depth Scaling | Exclusive Learning<br>InclusiveFL<br>DepthFL (FedAvg) | 67.08<br>47.61<br>66.18 | 68.00<br>53.88<br>67.56 | 66.19<br>59.48<br>67.97 | 56.78<br>60.46<br>68.01 |

```bash
# table 4
python -m depthfl.main --multirun fit_config.kd=true,false dataset_config.iid=true,false
```

**Table 4** 

Accuracy of the global model with/without self distillation on CIFAR-100.

| Distribution | Dataset | KD | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 | Ensemble | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IID | CIFAR100 | &cross;<br>&check; | 70.13<br>71.74 | 69.63<br>73.35 | 68.92<br>73.57 | 68.92<br>73.55 | 74.48<br>76.06 | 
| non-IID | CIFAR100 | &cross;<br>&check; | 67.94<br>70.33 | 68.68<br>71.88 | 68.46<br>72.43 | 67.78<br>72.34 | 73.18<br>74.92 | 

