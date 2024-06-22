# Flower Datasets

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/actions/workflows/framework.yml/badge.svg)
![Downloads](https://pepy.tech/badge/flwr-datasets)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.ai/join-slack)

Flower Datasets (`flwr-datasets`) is a library to quickly and easily create datasets for federated learning, federated evaluation, and federated analytics. It was created by the `Flower Labs` team that also created Flower: A Friendly Federated Learning Framework.
For complete documentation that includes API docs, how-to guides and tutorials please visit https://flower.ai/docs/datasets/ and for full FL example see https://github.com/adap/flower/tree/main/examples.
Below you'll find a brief overview of the library.

Flower Datasets library supports:
* **downloading datasets** - choose the dataset from Hugging Face's `datasets`,
* **partitioning datasets** - customize the partitioning scheme,
* **creating centralized datasets** - leave parts of the dataset unpartitioned (e.g. for centralized evaluation).

Thanks to using Hugging Face's `datasets` used under the hood, Flower Datasets integrates with the following popular formats/frameworks:
* Hugging Face,
* PyTorch,
* TensorFlow,
* Numpy,
* Pandas,
* Jax,
* Arrow.

Create **custom partitioning schemes** or choose from the **implemented partitioning schemes**:

* Partitioner (the abstract base class) `Partitioner`
* IID partitioning `IidPartitioner(num_partitions)`
* Dirichlet partitioning `DirichletPartitioner(num_partitions, partition_by, alpha)`
* InnerDirichlet partitioning `InnerDirichletPartitioner(partition_sizes, partition_by, alpha)`
* Natural ID partitioner `NaturalIdPartitioner(partition_by)`
* Size partitioner (the abstract base class for the partitioners dictating the division based the number of samples) `SizePartitioner`
* Linear partitioner `LinearPartitioner(num_partitions)`
* Square partitioner `SquarePartitioner(num_partitions)`
* Exponential partitioner `ExponentialPartitioner(num_partitions)`
* more to come in the future releases (contributions are welcome).
<p align="center">
  <img src="./doc/source/_static/readme/comparison_of_partitioning_schemes.png" alt="Comparison of partitioning schemes."/>
  <br>
  <em>Comparison of Partitioning Schemes on CIFAR10</em>
</p>
PS: This plot was generated using a library function (see flwr_datasets.visualization package for more).

# Installation

## With pip

Flower Datasets can be installed from PyPi

```bash
pip install flwr-datasets
```

Install with an extension:

* for image datasets:

```bash
pip install flwr-datasets[vision]
```

* for audio datasets:

```bash
pip install flwr-datasets[audio]
```

If you plan to change the type of the dataset to run the code with your ML framework, make sure to have it installed too.

# Usage

Flower Datasets exposes the `FederatedDataset` abstraction to represent the dataset needed for federated learning/evaluation/analytics. It has two powerful methods that let you handle the dataset preprocessing: `load_partition(partition_id, split)` and `load_split(split)`.

Here's a basic quickstart example of how to partition the MNIST dataset:

```
from flwr_datasets import FederatedDataset

# The train split of the MNIST dataset will be partitioned into 100 partitions
fds = FederatedDataset("mnist", partitioners={"train": 100}

partition = fds.load_partition(0)

centralized_data = fds.load_split("test")
```

For more details, please refer to the specific how-to guides or tutorial. They showcase customization and more advanced features.

# Future release

Here are a few of the things that we will work on in future releases:

* ✅ Support for more datasets (especially the ones that have user id present).
* ✅ Creation of custom `Partitioner`s.
* ✅ More out-of-the-box `Partitioner`s.
* ✅ Passing `Partitioner`s via `FederatedDataset`'s `partitioners` argument.
* ✅ Customization of the dataset splitting before the partitioning.
* ✅ Simplification of the dataset transformation to the popular frameworks/types.
* Creation of the synthetic data,
* Support for Vertical FL.
