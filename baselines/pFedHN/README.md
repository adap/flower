---
title: Personalized Federated Learning using Hypernetworks
url: https://arxiv.org/abs/2103.04628
labels: ["data heterogenity", "hypernetworks","personalised federated learning",] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [MNIST, CIFAR10, CIFAR100] # list of datasets you include in your baseline
---

# *pFedHN*



****Paper:**** :https://arxiv.org/abs/2103.04628

****Authors:**** :Aviv Shamsian,Aviv Navon,Ethan Fetaya,Gal Chechik

****Abstract:**** :Personalized federated learning is tasked with training machine learning models for multiple clients, each with its own data distribution. The goal is to train personalized models in a collaborative way while accounting for data disparities across clients and reducing communication costs. We propose a novel approach to this problem using hypernetworks, termed pFedHN for personalized Federated HyperNetworks. In this approach, a central hypernetwork model is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients, while maintaining the capacity to generate unique and diverse personal models. Furthermore, since hypernetwork parameters are never transmitted, this approach decouples the communication cost from the trainable model size. We test pFedHN empirically in several personalized federated learning challenges and find that it outperforms previous methods. Finally, since hypernetworks share information across clients we show that pFedHN can generalize better to new clients whose distributions differ from any client observed during training.

## About this baseline

****What’s implemented:**** :"The code in the repository reproduces the paper by implementing the concept of hypernetworks which create weights for that target network thus resolving the problems of dataheterogenity.The hypernetworks lies in the server and the clients have the target model. Initially hypernetwork sends the weights from the server which is loaded into the target net . After the targetnet is trained we pass the delta_theta to the client , which inturn updated the phi_gradients for the hypernetwork and the flow goes on."

****Datasets:**** :[MNIST,CIFAR10,CIFAR100]

****Hardware Setup:**** : Will be updated

****Contributors:**** :*_Ram Samarth B B(@achiverram28) , Kishan Gurumurthy(@kishan-droid) , Sachin DN(@sachugowda)_*


## Experimental Setup

****Task:**** : Image Classification

****Model:**** : CNNHyper for the HyperNetwork , CNNTarget For the TargetNetwork

**Dataset:** This baseline includes the MNIST, CIFAR10 , CIFAR100 datasets. By default it will be partitioned into 50 clients following IID distribution. The settings are as follow:

| Dataset | #classes | partitioning method |
| :------ | :---: | :---: | :---: |
| MNIST | 10 | IID |
| CIFAR10 | 10 | IID |
| CIFAR10 | 100 | IID |


****Training Hyperparameters:**** : The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python3 main.py` directly)

| Description | Default Value |
| ----------- | ----- |
| total clients | 50 |
| local epoch | 50 |
| batch size | 64 |
| number of rounds | 5000 |
| HyperNetwork learning rate | 1e-2 |
| TargetNetwork learning rate | 5e-3 |
| TargetNetwork weight decay | 5e-5 |
| TargetNetwork momentum | 0.9 |
| data partition | IID |
| HyperNetwork Optimizer | Adam |
| TargetNetwork Optimizer | SGD with weight decay |
**Model variations** : The model is flexible to use for images with 1 channel
and 28x28 dimension , images with 3 channels and 32X32 dimension(for MNIST,CIFAR10 and CIFAR)
| Default channels for the target CNN | 3  |
| Default outclasses for the target CNN | 10 |
| Default kernels for the target CNN | 16 |



## Environment Setup


To construct the Python environment follow these steps:

```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

To run this pFedHN, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
python3 -m pFedHN.main # this will run using the default settings in the `conf/config.yaml` that is for the cifar10

python3 -m pFedHN.main dataset.data="mnist" model.n_kernels=7 model
.in_channels=1 # this will run for the mnist

python3 -m pFedHN.main dataset.data="cifar100" model.out_dim=100 # this will run for the cifar100

```

## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._
