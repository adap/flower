---
Title: Personalized Federated Learning using Hypernetworks
Url: https://arxiv.org/abs/2103.04628
Labels: [data heterogenity, hypernetworks, personalised FL,]
Dataset: [MNIST, CIFAR-10, CIFAR-100]
---

# Personalized Federated Learning using Hypernetworks

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2103.04628](https://arxiv.org/abs/2103.04628)

**Authors:** Aviv Shamsian, Aviv Navon, Ethan Fetaya, Gal Chechik

**Abstract:** Personalized federated learning is tasked with training machine learning models for multiple clients, each with its own data distribution. The goal is to train personalized models in a collaborative way while accounting for data disparities across clients and reducing communication costs. We propose a novel approach to this problem using hypernetworks, termed pFedHN for personalized Federated HyperNetworks. In this approach, a central hypernetwork model is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients, while maintaining the capacity to generate unique and diverse personal models. Furthermore, since hypernetwork parameters are never transmitted, this approach decouples the communication cost from the trainable model size. We test pFedHN empirically in several personalized federated learning challenges and find that it outperforms previous methods. Finally, since hypernetworks share information across clients we show that pFedHN can generalize better to new clients whose distributions differ from any client observed during training.

## About this baseline

**What’s implemented:** The code in the repository reproduces the paper by implementing the concept of hypernetworks which create weights for that target network thus resolving the problems of dataheterogenity. The hypernetworks lies in the server and the clients have the target model. Initially hypernetwork sends the weights from the server which is loaded into the target net. After the targetnet is trained we pass the delta_theta to the client, which inturn updated the phi_gradients for the hypernetwork and the flow goes on.

**Datasets:** MNIST, CIFAR-10, CIFAR-100 from torchvision 

**Hardware Setup:** The experiments were conducted on a 12-core CPU MacBook Pro M2 Pro with 32GB of RAM, as well as on an HPC Cluster equipped with NVIDIA A100-PCIE-40GB GPU, alternately.

**Contributors:** 
---
| Names     | Profiles |
| ----------- | ----------- |
| Ram Samarth B B      | [achiverram28](https://github.com/achiverram28)      |
| Kishan Gurumurthy   | [kishan-droid](https://github.com/kishan-droid)      |
| Sachin DN | [sachugowda](https://github.com/sachugowda) |
---

## Experimental Setup

**Task:** Image Classification

**Model:** CNNHyper for the HyperNetwork , CNNTarget For the TargetNetwork

**Dataset:** This baseline includes the MNIST, CIFAR-10 , CIFAR-100 datasets. By default it will be partitioned into 50 clients following Non-IID distribution. The settings are as follow:

| Dataset | #classes | partitioning method | classes per client |
| :------ | :---: | :---: | :---: |
| MNIST | 10 | Non-IID | 2 |
| CIFAR10 | 10<br>100 | Non-IID | 2<br>10 |


**Training Hyperparameters:** The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `python3 -m pFedHN.main` directly)

| Description | Default Value |
| ----------- | ----- |
| Data Partition | Non - IID |
| Dataset | CIFAR-10 |
| Batch size | 64 |
| Classes per client | 2|
| Total clients | 50 |
| Tocal epoch(client-side) | 50 |
| Number of rounds | 5000 |
| HyperNetwork hidden units for CIFAR-10/CIFAR-100 | 3 |
| HyperNetwork hidden units for MNIST | 1 |
| HyperNetwork learning rate | 1e-2 |
| HyperNetwork momentum | 0.9 |
| HyperNetwork weight decay | 1e-3 |
| HyperNetwork Optimizer | SGD with momentum and weight decay |
| TargetNetwork learning rate | 5e-3 |
| TargetNetwork momentum | 0.9 |
| TargetNetwork weight decay | 5e-5 |
| TargetNetwork Optimizer | SGD with momentum and weight decay |
**Target Model Variations** 
| **Dataset** | CIFAR-10 |
| Number of input channels | 3 |
| Input Image Dimension | 32x32 |
| Number of classes | 10 |
| Kernels | 16 |
| **Dataset** | CIFAR-100 |
| Number of input channels | 3 |
| Input Image Dimension | 32x32 |
| Number of classes | 100 |
| Kernels | 16 |
| **Dataset** | MNIST |
| Number of input channels | 1 |
| Input Image Dimension | 28x28 |
| Number of classes | 10 |
| Kernels | 7 |
| Local Layer for pFedHN | False |
| Local Layer for pFedHNPC | True |
| Learning rate for pFedHNPC | 5e-2 |




## Environment Setup


To construct the Python environment follow these steps:

```bash
# Set Python 3.10
pyenv local 3.10.11

# Tell poetry to use python 3.10
poetry env use 3.10.11

# Install the base Poetry environment
poetry install

# Activate the environment
poetry shell
```

## Running the Experiments

To run this pFedHN, first ensure you have activated your Poetry environment (execute `poetry shell` from this directory), then:

```bash
## These instructions are for the pFedHN Experiments
python -m pFedHN.main # this will run using the default settings in the `conf/config.yaml` that is for the CIFAR-10 dataset

python -m pFedHN.main dataset.data="cifar100" model.out_dim=100 client.num_classes_per_node=10 # this will run for the CIFAR-100 dataset where we give each client 10 classes and number of classes is 100

python -m pFedHN.main dataset.data="mnist" model.n_kernels=7 model.in_channels=1 model.n_hidden=1 # this will run for the MNIST dataset where the number of input channels is 1 , the number of hidden layers in hypernetwork is 1 and the number of kernels used in the CNNTarget is 7

## For conducting pFedHNPC Experiments follow the commands given below

# For MNIST we are not conducting experiments as the paper has not done it.

python -m pFedHN.main model.local=True model.variant=1 server.lr=5e-2 # this will run the pFedHNPC for CIFAR-10 dataset where local=True is for using LocalLayer and variant=1 for setting pFedHNPC . Learning rate is modified to 5e-2

python -m pFedHN.main dataset.data="cifar100" model.out_dim=100 client.num_classes_per_node=10 model.local=True model.variant=1 server.lr=5e-2 # this will run the pFedHNPC for CIFAR-100 dataset where local=True is for using LocalLayer and variant=1 for setting pFedHNPC . Learning rate is modified to 5e-2

```

## Expected Results

| Algorithm | Dataset | Num_Clients | Paper_Accuracy | Experimented_Accuracy | Experimented_Loss | Hardware | Time-Taken |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| pFedHN | MNIST | 50 | 99.28 ± 0.11 | 99.18 | 0.04258 | NVIDIA A100-PCIE-40GB, num_gpus=0.5 |  9602.4372 seconds |
| pFedHN | CIFAR-10 | 50 | 88.38 ± 0.29 | 82.37 | 0.81694 | MacBook Pro M2 Pro, 12 Core CPU | 15252.1581 seconds |
| pFedHNPC | CIFAR-10 | 50 | 90.08 ± 0.63 | 85.25 | 0.70374 | MacBook Pro M2 Pro, 12 Core CPU | 15279.2597 seconds |
