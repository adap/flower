---
title: "Federated Learning on Non-IID Data Silos: An Experimental Study"
url: https://arxiv.org/abs/2102.02079
labels: [data heterogeneity, image classification, benchmark]
dataset: [CIFAR-10, MNIST, Fashion-MNIST]
algorithms: [FedAvg, SCAFFOLD, FedProx, FedNova]
---

# Federated Learning on Non-IID Data Silos: An Experimental Study

**Paper:** [arxiv.org/abs/2102.02079](https://arxiv.org/abs/2102.02079)

**Authors:** Qinbin Li, Yiqun Diao, Quan Chen, Bingsheng He

**Abstract:** Due to the increasing privacy concerns and data regulations, training data have been increasingly fragmented, forming distributed databases of multiple "data silos" (e.g., within different organizations and countries). To develop effective machine learning services, there is a must to exploit data from such distributed databases without exchanging the raw data. Recently, federated learning (FL) has been a solution with growing interests, which enables multiple parties to collaboratively train a machine learning model without exchanging their local data. A key and common challenge on distributed databases is the heterogeneity of the data distribution among the parties. The data of different parties are usually non-independently and identically distributed (i.e., non-IID). There have been many FL algorithms to address the learning effectiveness under non-IID data settings. However, there lacks an experimental study on systematically understanding their advantages and disadvantages, as previous studies have very rigid data partitioning strategies among parties, which are hardly representative and thorough. In this paper, to help researchers better understand and study the non-IID data setting in federated learning, we propose comprehensive data partitioning strategies to cover the typical non-IID data cases. Moreover, we conduct extensive experiments to evaluate state-of-the-art FL algorithms. We find that non-IID does bring significant challenges in learning accuracy of FL algorithms, and none of the existing state-of-the-art FL algorithms outperforms others in all cases. Our experiments provide insights for future studies of addressing the challenges in "data silos".


## About this baseline

**What’s implemented:** The code in this directory replicates many experiments from the aforementioned paper. Specifically, it contains implementations for four FL protocols, `FedAvg` (McMahan et al. 2017), `SCAFFOLD` (Karimireddy et al. 2019), `FedProx` (Li et al. 2018), and `FedNova` (Wang et al. 2020). The FL protocols are evaluated across various non-IID data partition strategies across clients on three image classification datasets MNIST, CIFAR10, and Fashion-mnist.

**Datasets:** MNIST, CIFAR10, and Fashion-mnist from PyTorch's Torchvision

**Hardware Setup:** These experiments were run on a linux server with 56 CPU threads with 250 GB Ram. There are 105 configurations to run per seed and at any time 7 configurations have been run parallelly. The experiments required close to 12 hrs to finish for one seed. Nevertheless, to run a subset of configurations, such as only one FL protocol across all datasets and splits, a machine with 4-8 threads and 16 GB memory can run in reasonable time.

**Contributors:** Aashish Kolluri, PhD Candidate, National University of Singapore


## Experimental Setup

**Task:** Image classification

**Model:** This directory implements CNNs as mentioned in the paper (Section V, paragraph 1). Specifically, the CNNs have two 2D convolutional layers with 6 and 16 output channels, kernel size 5, and stride 1.

**Dataset:** This directory has three image classification datasets that are used in the baseline, MNIST, CIFAR10, and Fashion-mnist. Further, five different data-splitting strategies are used including iid and four non-iid strategies based on label skewness. In the first non-iid strategy, for each label the data is split based on proportions sampled from a dirichlet distribution (with parameter 0.5). In the three remaining strategies, each client gets data from randomly chosen #C labels where #C is 1, 2, or 3. For the clients that are supposed to receive data from the same label the data is equally split between them. The baseline considers 10 clients. The following table shows all dataset and data splitting configurations.

| Datasets | #classes | #partitions | partitioning method | partition settings |
| :------ | :---: | :---: | :---: | :---: |
| CIFAR10, MNIST, Fashion-mnist | 10 | 10 | IID<br>dirichlet<br>sort and partition<br>sort and partition<br>sort and partition | NA<br>distribution parameter 0.5<br>1 label per client<br>2 labels per client<br>3 labels per client |


**Training Hyperparameters:** There are four FL algorithms and they have many common hyperparameters and few different ones. The following table shows the common hyperparameters and their default values.

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
# Setup the base poetry environment from the niid_bench directory
# Set python version
pyenv local 3.10.6
# Tell poetry to use python 3.10
poetry env use 3.10.6
# Now install the environment
poetry install
# Start the shell
poetry shell
```


## Running the Experiments
You can run four algorithms `FedAvg`, `SCAFFOLD`, `FedProx`, and `FedNova`. To run any of them, use any of the corresponding config files. For instance, the following command will run with the default config provided in the corresponding configuration files.

```bash
# Run with default config, it will run FedAvg on cpu-only mode
python -m niid_bench.main
# Below to enable GPU utilization by the server and the clients.
python -m niid_bench.main server_device=cuda client_resources.num_gpus=0.2
```

To change the configuration such as dataset or hyperparameters, specify them as part of the command line arguments.

```bash
python -m niid_bench.main --config-name scaffold_base dataset_name=mnist partitioning=iid # iid
python -m niid_bench.main --config-name fedprox_base dataset_name=mnist partitioning=dirichlet # dirichlet
python -m niid_bench.main --config-name fednova_base dataset_name=mnist partitioning=label_quantity labels_per_client=3 # sort and partition
```


## Expected Results

We provide the bash script run_exp.py that can be used to run all configurations. For instance, the following command runs all of them with 4 configurations running at the same time. Consider lowering `--num-processes` if your machine runs slow. With `--num-processes 1` one experiment will be run at a time.

```bash
python run_exp.py --seed 42 --num-processes 4
```

The above command generates results that can be parsed to get the best accuracies across all rounds for each configuration which can be presented in a table (similar to Table 3 in the paper).

| Dataset | partitioning method | FedAvg | SCAFFOLD | FedProx | FedNova |
| :------ | :------ | :---: | :---: | :---: | :---: |
| MNIST | IID<br>Dirichlet (0.5)<br>Sort and Partition (1)<br>Sort and Partition (2)<br>Sort and Partition (3) | 99.09 &#xB1; 0.05<br>98.89 &#xB1; 0.07<br>19.33 &#xB1; 11.82<br>96.86 &#xB1; 0.30<br>97.86 &#xB1; 0.34 | 99.06 &#xB1; 0.15<br>99.07 &#xB1; 0.06<br>9.93 &#xB1; 0.12<br>96.92 &#xB1; 0.52<br>97.91 &#xB1; 0.10 | 99.16 &#xB1; 0.04<br>99.02 &#xB1; 0.02<br>51.79 &#xB1; 26.75<br>96.85 &#xB1; 0.15<br>97.85 &#xB1; 0.06 | 99.05 &#xB1; 0.06<br>98.03 &#xB1; 0.06<br>52.58 &#xB1; 14.08<br>96.65 &#xB1; 0.39<br>97.62 &#xB1; 0.07 |
| FMNIST | IID<br>Dirichlet (0.5)<br>Sort and Partition (1)<br>Sort and Partition (2)<br>Sort and Partition (3) | 89.23 &#xB1; 0.45<br>88.09 &#xB1; 0.29<br>28.39 &#xB1; 17.09<br>78.10 &#xB1; 2.51<br>82.43 &#xB1; 1.52 | 89.33 &#xB1; 0.27<br>88.44 &#xB1; 0.25<br>10.00 &#xB1; 0.00<br>33.80 &#xB1; 41.22<br>80.32 &#xB1; 5.03 | 89.42 &#xB1; 0.09<br>88.15 &#xB1; 0.42<br>32.65 &#xB1; 6.68<br>78.05 &#xB1; 0.99<br>82.99 &#xB1; 0.48 | 89.36 &#xB1; 0.09<br>88.22 &#xB1; 0.12<br>16.86 &#xB1; 9.30<br>71.67 &#xB1; 2.34<br>81.97 &#xB1; 1.34 |
| CIFAR10 | IID<br>Dirichlet (0.5)<br>Sort and Partition (1)<br>Sort and Partition (2)<br>Sort and Partition (3) | 71.32 &#xB1; 0.33<br>62.47 &#xB1; 0.43<br>10.00 &#xB1; 0.00<br>51.17 &#xB1; 1.09<br>59.11 &#xB1; 0.87 | 71.66 &#xB1; 1.13<br>68.08 &#xB1; 0.96<br>10.00 &#xB1; 0.00<br>49.42 &#xB1; 2.18<br>61.00 &#xB1; 0.91 | 71.26 &#xB1; 1.18<br>65.63 &#xB1; 0.08<br>12.71 &#xB1; 0.96<br>50.44 &#xB1; 0.79<br>59.20 &#xB1; 1.18 | 70.69 &#xB1; 1.14<br>63.89 &#xB1; 1.40<br>10.00 &#xB1; 0.00 <br>46.9 &#xB1; 0.66<br>57.83 &#xB1; 0.42 |
