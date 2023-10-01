---
title: Personalized Federated Learning with Feature Alignment and Classifier Collaboration
url: https://openreview.net/forum?id=SXZr8aDKia
labels: [label1, label2] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [dataset1, dataset2] # list of datasets you include in your baseline
---

# Personalized Federated Learning with Feature Alignment and Classifier Collaboration

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://openreview.net/forum?id=SXZr8aDKia

****Authors:**** Jian Xu, Xinyi Tong, Shao-Lun Huang

****Abstract:**** Data heterogeneity is one of the most challenging issues in federated learning, which motivates a variety of approaches to learn personalized models for participating clients. One such approach in deep neural networks based tasks is employing a shared feature representation and learning a customized classifier head for each client. However, previous works do not utilize the global knowledge during local representation learning and also neglect the fine-grained collaboration between local classifier heads, which limits the model generalization ability. In this work, we conduct explicit local-global feature alignment by leveraging global semantic knowledge for learning a better representation. Moreover, we quantify the benefit of classifier combination for each client as a function of the combining weights and derive an optimization problem for estimating optimal weights. Finally, extensive evaluation results on benchmark datasets with various heterogeneous data scenarios demonstrate the effectiveness of our proposed method.


## About this baseline

****What’s implemented:**** The code in this directory replicates the experiments in *Personalized Federated Learning with Feature Alignment and Classifier Collaboration* (Xu et al., 2023) for EMNIST and CIFAR10, which proposed the FedPAC algorithm. Concretely, it replicates the results for EMNIST and CIFAR10 in Table 1.

****Datasets:**** EMNIST and CIFAR10 from PyTorch's Torchvision
<!-- 
****Hardware Setup:**** These experiments were run on a desktop machine with 24 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: we install PyTorch with GPU support but by default, the entire experiment runs on CPU-only mode. -->

****Contributors:**** Apsal S Kalathukunnel


## Experimental Setup

****Task:**** Image classification

****Model:**** This directory implements twp different CNN models
for EMNIST/Fashion-MNIST and CIFAR-10/CINIC-10, respectively. The first CNN model is constructed by two convolution layers with 16 and 32 channels respectively, each followed by a max pooling layer, and two fully-connected layers with 128 and 10 units before softmax output. LeakyReLU is used as the activation function. The second CNN model is similar to the first one but has one more convolution layer with 64 channels.

****Dataset:**** Two datasets used for experiments are EMNIST and CIFAR10. EMNIST (Extended MNIST) is a 62-class image classification dataset, extending the classic MNIST dataset. It contains 62 categories of handwritten characters, including 10 digits, 26
uppercase letters and 26 lowercase letters. CIFAR-10 with 10 categories of color images. In the experiments, all clients have the same data size, in which s% of data (20% by default) are uniformly sampled from all classes, and the remaining (100 - s)% from a set of dominant classes for each client. Clients are explicitly divided into multiple groups where clients in each group share the same dominant
classes, and we also intentionally keep the size of local training data small to pose the need for FL. The testing data on each client has the same distribution as the training data.



****Training Hyperparameters:**** The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python main.py` directly)

| Description | Default Value |
| ----------- | ----- |
| total clients | 100 |
| number of rounds | 200 |
| client resources | {'num_cpus': 2.0, 'num_gpus': 0.0 }|
| optimizer | SGD|



## Environment Setup

To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell

```


## Running the Experiments

To run this FedPAC baseline, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
python -m fedpac.main # this will run using the default settings in the `conf/config.yaml`

# you can override settings directly from the command line
python -m fedpac.main num_rounds=200 

# if you run this baseline with a larger model, you might want to use the GPU (not used by default).
# you can enable this by overriding the `server_device` and `client_resources` config. For example
# the below will run the server model on the GPU and 4 clients will be allowed to run concurrently on a GPU (assuming you also meet the CPU criteria for clients)
python -m fedpac.main server_device=cuda client_resources.num_gpus=0.25
```

To run using FedAvg:
```bash
# this will use a variation of FedAvg that drops the clients that were flagged as stragglers
# This is done so to match the experimental setup in the FedProx paper
python -m fedpac.main --config-name fedavg

# this config can also be overriden from the CLI
```

## Expected Results


