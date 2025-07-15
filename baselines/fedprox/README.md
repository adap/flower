---
title: Federated Optimization in Heterogeneous Networks
url: https://arxiv.org/abs/1812.06127
labels: [image classification, cross-device, stragglers]
dataset: [MNIST, FEMNIST]
---
# FedProx: Federated Optimization in Heterogeneous Networks

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/1812.06127](https://arxiv.org/abs/1812.06127)

**Authors:** Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith.

**Abstract:** Federated Learning is a distributed learning paradigm with two key challenges that differentiate it from traditional distributed optimization: (1) significant variability in terms of the systems characteristics on each device in the network (systems heterogeneity), and (2) non-identically distributed data across the network (statistical heterogeneity). In this work, we introduce a framework, FedProx, to tackle heterogeneity in federated networks. FedProx can be viewed as a generalization and re-parametrization of FedAvg, the current state-of-the-art method for federated learning. While this re-parameterization makes only minor modifications to the method itself, these modifications have important ramifications both in theory and in practice. Theoretically, we provide convergence guarantees for our framework when learning over data from non-identical distributions (statistical heterogeneity), and while adhering to device-level systems constraints by allowing each participating device to perform a variable amount of work (systems heterogeneity). Practically, we demonstrate that FedProx allows for more robust convergence than FedAvg across a suite of realistic federated datasets. In particular, in highly heterogeneous settings, FedProx demonstrates significantly more stable and accurate convergence behavior relative to FedAvg---improving absolute test accuracy by 22% on average.

## About this baseline

**What's implemented:** The code in this directory replicates the experiments in *Federated Optimization in Heterogeneous Networks* (Li et al., 2018) for MNIST and FEMNIST, which proposed the FedProx algorithm. Concretely, it replicates the results for both datasets in Figure 1 and 7.

**Datasets:** MNIST, FEMNIST

**Hardware Setup:** These experiments were run on a desktop machine with 24 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: we install PyTorch with GPU support but by default, the entire experiment runs on CPU-only mode.

**Contributors:** Charles Beauville, Javier Fernandez-Marques, Andrej Jovanović and Ashley Au

## Experimental Setup

**Task:** Image classification

**Model:** A logistic regression model used in the FedProx paper (see `model`). This is the model used by default.

**Dataset:** This baseline includes both MNIST and FEMNIST datasets. For MNIST, it will be partitioned into 1000 clients following a pathological split where each client has examples of two (out of ten) class labels. For FEMNIST, we follow the paper to subsample 10 lower case characters ('a'-'j') from EMNIST and distribute only 5 classes to each client. The number of examples in each client is derived by sampling from a powerlaw distribution. The settings are as follows:

| Dataset | #classes | #rounds | #partitions |     partitioning method     |  partition settings  |
| :------ | :------: | :-----: | :---------: | :-------------------------: | :------------------: |
| MNIST   |    10    |   100   |    1000    | pathological with power law | 2 classes per client |
| FEMNIST |    10    |   200   |     200     | pathological with power law | 5 classes per client |

**Training Hyperparameters:**
The following table shows the main hyperparameters for this baseline with their default value (i.e. the value used if you run `flwr run .` directly)

| Description         | Default Value                                      |
| ------------------- | -------------------------------------------------- |
| total clients       | 1000                                               |
| clients per round   | 10                                                 |
| number of rounds    | 100                                                |
| client resources    | {'num_cpus': 2.0, 'num_gpus': 0.0 }                |
| data partition      | pathological with power law (2 classes per client) |
| optimizer           | SGD with proximal term                             |
| proximal mu         | 1.0                                                |
| stragglers_fraction | 0.9                                                |

**Configurations:**

The following table shows the configurations to be set in `pyproject.toml` for different experiments

|  config.dataset.path  | config.dataset.num-unique-labels | config.dataset.num-unique-labels-per-partition | config.num-server-rounds | options.num-supernodes |
| :--------------------: | :------------------------------: | :--------------------------------------------: | :----------------------: | :--------------------: |
|   `"ylecun/mnist"`   |                10                |                       2                       |           100           |          1000          |
| `"flwrlabs/femnist"` |                10                |                       5                       |           200           |          200          |

## FEMNIST Preprocessor

According to the paper, the authors subsample 10 lower case characters ('a'-'j') from EMNIST and called this *federated* version of EMNIST as FEMNIST. Therefore, we constructed a Preprocessor class to achieve the above.

Note that EMNIST also does not have a test set to begin with, therefore we manually split the dataset using the configured `val_ratio` in the function `prepare_test_loader` to create the centralised test dataset.

## Environment Setup

To construct the Python environment, simply run:

```bash
# Set directory to use python 3.10 (install with `pyenv install <version>` if you don't have it)
pyenv virtualenv 3.10.14 fedprox

# Tell poetry to use python3.10
pyenv activate fedprox

# Install
pip install -e .
```

## Running the Experiments

To run this FedProx, first ensure you have activated your environment as above, then:

```bash
flwr run .  # this will run using the default settings in the `pyproject.toml`

# you can override settings directly from the command line
flwr run . --run-config "algorithm.mu=2 dataset.mu=2 algorithm.num_server_rounds=200" # will set proximal mu to 2 and the number of rounds to 200

# if you run this baseline with a larger model, you might want to use the GPU (not used by default).
# you can enable this by overriding the federation config. For example
# the below will run the server model on the GPU and 4 clients will be allowed to run concurrently on a GPU (assuming you also meet the CPU criteria for clients)
flwr run . gpu-simulation
```

To run using FedAvg:

```bash
# this will use a variation of FedAvg that drops the clients that were flagged as stragglers
# This is done so to match the experimental setup in the FedProx paper
flwr run . --run-config conf/mnist/fedavg_sf_0.9.toml  # MNIST dataset
flwr run . --run-config conf/femnist/fedavg_sf_0.9.toml  # FEMNIST dataset
```


## Expected results

With the following command, we run both FedProx and FedAvg configurations while iterating through different values of `mu` and `stragglers_fraction`. We ran each experiment five times to ensure that the results are significant

```bash
bash ./run_mnist_experiments.sh  # MNIST dataset
bash ./run_femnist_experiments.sh  # FEMNIST dataset
```

The configurations of the specific experiments within this one large ran can be found in the `conf` directory.

The above commands would generate results that you can plot and would look like the plots shown below. This plot was generated using the jupyter notebook in the `docs/` directory of this baseline after running the command above.

Results for MNIST:

![](_static/FedProx_mnist.png)

Results for FEMNIST:

![](_static/FedProx_femnist.png)
