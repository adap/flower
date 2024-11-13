---
title: FedRS: Federated Learning with Restricted Softmax for LabelDistribution Non-IID Data
url: https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf
labels: [data heterogeneity, image classification] 
dataset: [cifar10] 
---

# FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data

> [!NOTE]
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf

**Authors:** Xin-Chun Li, De-Chuan Zhan

**Abstract:** Federated Learning (FL) aims to generate a global shared model via collaborating decentralized clients with privacy considerations. Unlike standard distributed optimization, FL takes multiple optimization steps on local clients and then aggregates the model updates via a parameter server. Although this significantly reduces communication costs, the non-iid property across heterogeneous devices could make the local update diverge a lot, posing a fundamental challenge to aggregation. In this paper, we focus on a special kind of non-iid scene, i.e., label distribution skew, where each client can only access a partial set of the whole class set. Considering top layers of neural networks are more task-specific, we advocate that the last classification layer is more vulnerable to the shift of label distribution. Hence, we in-depth study the classifier layer and point out that the standard softmax will encounter several problems caused by missing classes. As an alternative, we propose “Restricted Softmax" to limit the update of missing classes’ weights during the local procedure. Our proposed FedRS is very easy to implement with only a few lines of code. We investigate our methods on both public datasets and a real-world service awareness application. Abundant experimental results verify the superiorities of our methods.

## About this baseline

**What’s implemented:** Performance comparison between `FedAvg`, `FedRS (alpha=0.5)` and `FedRS (alpha=0.9)` on VGG model and CIFAR10 with avg 5 classes per client (`Table 5` of paper)

**Datasets:** CIFAR10

**Hardware Setup:** The paper samples 10 clients for 1000 rounds. GPU is recommended. The results below were obtained on a machine with 1x NVIDIA L4 Tensor Core GPU, with 16 vCPUs and 64GB of RAM.

**Contributors:** [@flydump](https://github.com/flydump)

## Experimental Setup

**Task:** Image classification

**Model:** A 9-layer VGG model with 9,225,600 parameters and without batch norm. 

**Dataset:** The implementation is only for the CIFAR-10 dataset currently. Data is partitioned using [PathologicalPartitioner](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.PathologicalPartitioner.html) which determines how many classes will be assigned to each client. By default, the number of clients (i.e. partitions) is 100 and during each training round, 10% of clients are selected. Each client data contains 5 classes.

The paper uses a slightly different partitioning scheme, where samples for each class is divided into equal number of partitions, and allocate N shards (across all classes) to each client, such that each client has 5 classes on average.

**Training Hyperparameters:** Table below shows the training hyperparams for the experiments. Values are from the original paper where provided (e.g. learning rate, momentum, weight decay)

| Description | Default Value |
| ----------- | ----- |
| strategy | fedavg |
| scaling factor for missing classes (alpha) | 0.9 |
| fraction fit | 0.1 |
| local epochs | 2 |
| learning rate | 0.03 |
| momentum | 0.9 |
| weight decay | 5e-4 |
| number of rounds | 1000 |
| batch size | 64 |
| num classes per partition | 5 |
| model | vgg11 |

## Environment Setup


```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 <name-of-your-baseline-env>

# Activate it
pyenv activate <name-of-your-baseline-env>

# Install the baseline
pip install -e .
```

## Running the Experiments

## Table 5

The following three commands will generate the results for CIFAR10-100-5 in Table 5:

```bash
# CIFAR-10 (5 classes per client) with FedAvg
flwr run . --run-config conf/cifar10_100_5_fedavg.toml

# CIFAR-10 (5 classes per client) with FedRS (alpha=0.9)
flwr run . --run-config conf/cifar10_100_5_fedrs_0.9.toml

# CIFAR-10 (5 classes per client) with FedRS (alpha=0.5)
flwr run . --run-config conf/cifar10_100_5_fedrs_0.5.toml
```


The expected accuracy results are as follows:

|  | CIFAR10-100-5 
| ----------- | ----- |
| FedAvg | 0.810 | 
| FedRS ($\alpha$=0.5)| 0.837 | 
| FedRS ($\alpha$=0.9) | 0.840 | 
