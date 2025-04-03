## Overview

This introductory example to Flower uses PyTorch but deep knowledge of this frameworks is required to run the example. However, it will help you understand how to adapt Flower to your use case.
This example uses a partitioned CIFAR-10 dataset designed to test the performance of federated learning. There is artificial skew in the data distribution across nodes. For example, data on one node contains mostly classes 0-4, while data on the other node contains mostly classes 5-9.


## Quickstart

## Set up the project

### Hardware requirements

A NVIDIA GPU is strongly recommended. The training does not use more than 4GB of VRAM.

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/serverless-pytorch . \
        && rm -rf _tmp \
        && cd serverless-pytorch
```

This will create a new directory called `serverless-pytorch`.

### Install dependencies and project

Install the `flwr` on the serverless branch:

```bash
pip install git+https://github.com/zzsi/flower.git@serverless
```

## Run the project

```bash
python run_simulation.py
```

You should expect to see output like the following:

```
***** Starting experiment with config: {'project': 'cifar10', 'epochs': 20, 'batch_size': 128, 'steps_per_epoch': 1200,...


====== Starting Federated Learning Experiment ======
Configuration:
- Nodes: 2
- Strategy: fedavg
- Mode: Asynchronous
- Data split: skewed
- Rounds: 20
- Device: cuda
- Random seed: 0

----- Data Splitting Phase -----
Splitting data using 'skewed' strategy
Creating skewed partition with factor 0.5
splitted_classes [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]

=== Data Distribution Across Partitions ===
Skew Factor: 0.50
Number of Partitions: 2

Class Distribution by Partition:
--------------------------------------------------

Partition 0 (Total samples: 25105)
------------------------------
Class  |  Count  |  Percentage
------------------------------
   0   |   3779  |   15.05%
   1   |   3787  |   15.08%
   2   |   3744  |   14.91%
   3   |   3701  |   14.74%
   4   |   3737  |   14.89%
   5   |   1269  |    5.05%
   6   |   1275  |    5.08%
   7   |   1239  |    4.94%
   8   |   1290  |    5.14%
   9   |   1284  |    5.11%

Partition 1 (Total samples: 24895)
------------------------------
Class  |  Count  |  Percentage
------------------------------
   0   |   1221  |    4.90%
   1   |   1213  |    4.87%
   2   |   1256  |    5.05%
   3   |   1299  |    5.22%
   4   |   1263  |    5.07%
   5   |   3731  |   14.99%
   6   |   3725  |   14.96%
   7   |   3761  |   15.11%
   8   |   3710  |   14.90%
   9   |   3716  |   14.93%

==================================================
x_test shape: (10000, 32, 32, 3)
y_test shape: (10000,)

----- Starting Federated Training -----
Training type: concurrent
Number of rounds: 20
Steps per epoch: 1200
Batch size: 128
Training federated models concurrently using ThreadPoolExecutor
[Node 0] Epoch [1/20], Step [100/197], Loss: 2.142, Acc: 32.219%
[Node 1] Epoch [1/20], Step [100/195], Loss: 2.158, Acc: 32.320%
INFO :      node e36d79be-f9ec-4f94-89c4-deb7c36078c2 @ epoch 0: updating model weights using federated learning.
INFO :      node e36d79be-f9ec-4f94-89c4-deb7c36078c2 @ epoch 0: 195 local examples contributed to the last model update.
INFO :      node e36d79be-f9ec-4f94-89c4-deb7c36078c2 @ epoch 0: found 0 models from other nodes
INFO :      node a92db8de-ff36-4aba-973f-a30d3124d583 @ epoch 0: updating model weights using federated learning.
INFO :      node a92db8de-ff36-4aba-973f-a30d3124d583 @ epoch 0: 197 local examples contributed to the last model update.
INFO :      node a92db8de-ff36-4aba-973f-a30d3124d583 @ epoch 0: found 1 models from other nodes
WARNING :   No fit_metrics_aggregation_fn provided
INFO :      Aggregated metrics: {'num_examples': 392, 'num_nodes': 2, 'loss': 1.4144757049424308, 'accuracy': 0.3833446919962851}
```

## What's happening?

The simulation will run for 20 rounds. At each round, the model will be trained on the local dataset of each node. Then, the model weights will be aggregated across nodes using the FedAvg strategy.

