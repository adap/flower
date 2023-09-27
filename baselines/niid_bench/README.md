---
title: "Federated Learning on Non-IID Data Silos: An Experimental Study"
url: https://arxiv.org/abs/2102.02079
labels: [data heterogeneity, image classification, benchmark]
dataset: [cifar10, mnist, fashion-mnist]
algorithms: [fedavg, scaffold, fedprox, fednova]
---

# Federated Learning on Non-IID Data Silos: An Experimental Study

****Paper:**** : [https://arxiv.org/abs/2102.02079](https://arxiv.org/abs/2102.02079)

****Authors:**** : Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He

****Abstract:**** : Due to the increasing privacy concerns and data regulations, training data have been increasingly fragmented, forming distributed databases of multiple "data silos" (e.g., within different organizations and countries). To develop effective machine learning services, there is a must to exploit data from such distributed databases without exchanging the raw data. Recently, federated learning (FL) has been a solution with growing interests, which enables multiple parties to collaboratively train a machine learning model without exchanging their local data. A key and common challenge on distributed databases is the heterogeneity of the data distribution among the parties. The data of different parties are usually non-independently and identically distributed (i.e., non-IID). There have been many FL algorithms to address the learning effectiveness under non-IID data settings. However, there lacks an experimental study on systematically understanding their advantages and disadvantages, as previous studies have very rigid data partitioning strategies among parties, which are hardly representative and thorough. In this paper, to help researchers better understand and study the non-IID data setting in federated learning, we propose comprehensive data partitioning strategies to cover the typical non-IID data cases. Moreover, we conduct extensive experiments to evaluate state-of-the-art FL algorithms. We find that non-IID does bring significant challenges in learning accuracy of FL algorithms, and none of the existing state-of-the-art FL algorithms outperforms others in all cases. Our experiments provide insights for future studies of addressing the challenges in "data silos".


## About this baseline

****What’s implemented:**** The code in this directory replicates many experiments from the aforementioned paper. Specifically, it contains implementations for four FL protocols, FedAvg (McMahan et al. 2017), SCAFFOLD (Karimireddy et al. 2019), FedProx (Li et al. 2018), and FedNova (Wang et al. 2020). The FL protocols are evaluated across various non-IID data partition strategies across clients on three image classification datasets MNIST, CIFAR10, and Fashion-mnist.

****Datasets:**** MNIST, CIFAR10, and Fashion-mnist from PyTorch's Torchvision

****Hardware Setup:**** These experiments were run on a linux server with 56 CPU threads with 250 GB Ram. There are 105 configurations to run per seed and at any time 7 configurations have been run parallely. The experiments required close to 12 hrs to finish for one seed. Nevertheless, to run a subset of configurations, such as only one FL protocol across all datasets and splits, a machine with 4-8 threads and 16 GB memory can run in reasonable time.

****Contributors:**** Aashish Kolluri, PhD Candidate, National University of Singapore


## Experimental Setup

****Task:**** Image classification

****Model:**** This directory implements CNNs as mentioned in the paper (Section V, paragraph 1). Specifically, the CNNs have two 2D convolutional layers with 6 and 16 output channels, kernel size 5, and stride 1.

****Dataset:**** This directory has three image classification datasets that are used in the baseline, MNIST, CIFAR10, and Fashion-mnist. Further, five different data-splitting strategies are used including iid and four non-iid strategies based on label skewness. In the first non-iid strategy, for each label the data is split based on proportions sampled from a dirichlet distribution (with parameter 0.5). In the three remaining strategies, each client gets data from randomly chosen #C labels where #C is 1, 2, or 3. For the clients that are supposed to receive data from the same label the data is equally split between them. The baseline considers 10 clients. The following table shows all dataset and data splitting configurations.

| Dataset | #classes | #partitions | partitioning method | partition settings |
| :------ | :---: | :---: | :---: | :---: |
| CIFAR10 | 10 | 10 | IID | NA |
| CIFAR10 | 10 | 10 | dirichlet | distribution parameter 0.5 |
| CIFAR10 | 10 | 10 | sort and partition | 1 label per client |
| CIFAR10 | 10 | 10 | sort and partition | 2 labels per client |
| CIFAR10 | 10 | 10 | sort and partition | 3 labels per client |
| MNIST | 10 | 10 | IID | NA |
| MNIST | 10 | 10 | dirichlet | distribution parameter 0.5 |
| MNIST | 10 | 10 | sort and partition | 1 label per client |
| MNIST | 10 | 10 | sort and partition | 2 labels per client |
| MNIST | 10 | 10 | sort and partition | 3 labels per client |
| FMNIST | 10 | 10 | IID | NA |
| FMNIST | 10 | 10 | dirichlet | distribution parameter 0.5 |
| FMNIST | 10 | 10 | sort and partition | 1 label per client |
| FMNIST | 10 | 10 | sort and partition | 2 labels per client |
| FMNIST | 10 | 10 | sort and partition | 3 labels per client |


****Training Hyperparameters:**** There are four FL algorithms and they have many common hyperparameters and few different ones. The following table shows the common hyperparameters and their default values.

| Description | Default Value |
| ----------- | ----- |
| total clients | 10 |
| clients per round | 10 |
| number of rounds | 50 |
| number of local epochs | 10 |
| client resources | {'num_cpus': 4.0, 'num_gpus': 0.0 }|
| dataset name | cifar10 
| data partition | Dirichlet (0.5) |
| batch size | 64 |
| momentum for SGD | 0.9 |

For FedProx algorithm the proximal parameter is tuned from values {0.001, 0.01, 0.1, 1.0} in the experiments. The default value is 0.01. 


## Environment Setup

```bash
# Setup the base poetry enviroment from the niid_bench directory
poetry install

# Start the shell
poetry shell

#  Install tqdm, torch, and torchvision using pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm
```


## Running the Experiments
You can run four algorithms fedavg, scaffold, fedprox, and fednova. To run any of them, use any of the corresponding main files. For instance, the following command will run with the default config provided in the corresponding configuration files.

```bash
python -m niid_bench.main_fedprox
```

To change the configuration such as dataset or hyperparameters, specify them as part of the command line arguments.

```bash
python -m niid_bench.main_scaffold dataset_name=mnist partitioning=iid # iid
python -m niid_bench.main_fedavg dataset_name=mnist partitioning=dirichlet # dirichlet
python -m niid_bench.main_fednova dataset_name=mnist partitioning=label_quantity labels_per_client=3 # sort and partition
```


## Expected Results

We provide the bash script run_all.sh that can be used to run all configurations for one seed with 7 configurations running at the same time.

```bash
./run_all.sh
```

The above command generates results that can be parsed to get the accuracies for each configuration which can be presented in a table (similar to Table 3 in the paper).

| Dataset | partitioning method | FedAvg | SCAFFOLD | FedProx | FedNova |
| :------ | :------ | :---: | :---: | :---: | :---: |
| MNIST | IID | | | | |
| MNIST | Dirichlet (0.5) | | | | |
| MNIST | Sort and Partition (1) | | | | |
| MNIST | Sort and Partition (2) | | | | |
| MNIST | Sort and Partition (3) | | | | |
| FMNIST | IID | | | | |
| FMNIST | Dirichlet (0.5) | | | | |
| FMNIST | Sort and Partition (1) | | | | |
| FMNIST | Sort and Partition (2) | | | | |
| FMNIST | Sort and Partition (3) | | | | |
| CIFAR10 | IID | | | | |
| CIFAR10 | Dirichlet (0.5) | | | | |
| CIFAR10 | Sort and Partition (1) | | | | |
| CIFAR10 | Sort and Partition (2) | | | | |
| CIFAR10 | Sort and Partition (3) | | | | |

