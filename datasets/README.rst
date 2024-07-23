Flower Datasets
===============

.. image:: https://img.shields.io/github/license/adap/flower
    :target: https://github.com/adap/flower/blob/main/LICENSE
.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
    :target: https://github.com/adap/flower/blob/main/CONTRIBUTING.md
.. image:: https://github.com/adap/flower/actions/workflows/framework.yml/badge.svg
.. image:: https://pepy.tech/badge/flwr-datasets
.. image:: https://img.shields.io/badge/Chat-Slack-red
    :target: https://flower.ai/join-slack

Flower Datasets (``flwr-datasets``) is a library to quickly and easily create datasets for federated learning, federated evaluation, and federated analytics. It was created by the ``Flower Labs`` team that also created Flower: A Friendly Federated Learning Framework.

.. tip::
    For complete documentation that includes API docs, how-to guides and tutorials please visit the `Flower Datasets Documentation <https://flower.ai/docs/datasets/>`_ and for full FL example see the `Flower Examples page <https://github.com/adap/flower/tree/main/examples>`_.

Installation
------------

For a complete installation guide visit the `Flower Datasets Documentation <https://flower.ai/docs/datasets/>`_

.. code-block:: bash

    pip install flwr-datasets[vision]

Overview
--------

Flower Datasets library supports:

* **downloading datasets** - choose the dataset from Hugging Face's ``datasets``,
* **partitioning datasets** - customize the partitioning scheme,
* **creating centralized datasets** - leave parts of the dataset unpartitioned (e.g. for centralized evaluation).

Thanks to using Hugging Face's ``datasets`` used under the hood, Flower Datasets integrates with the following popular formats/frameworks:

* Hugging Face,
* PyTorch,
* TensorFlow,
* Numpy,
* Pandas,
* Jax,
* Arrow.

Create **custom partitioning schemes** or choose from the **implemented `partitioning schemes <https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html#module-flwr_datasets.partitioner>`_**:

* Partitioner (the abstract base class) ``Partitioner``
* IID partitioning ``IidPartitioner(num_partitions)``
* Dirichlet partitioning ``DirichletPartitioner(num_partitions, partition_by, alpha)``
* InnerDirichlet partitioning ``InnerDirichletPartitioner(partition_sizes, partition_by, alpha)``
* Pathological partitioning ``PathologicalPartitioner(num_partitions, partition_by, num_classes_per_partition, class_assignment_mode)``
* Natural ID partitioning ``NaturalIdPartitioner(partition_by)``
* Size based partitioning (the abstract base class for the partitioners dictating the division based the number of samples) ``SizePartitioner``
* Linear partitioning ``LinearPartitioner(num_partitions)``
* Square partitioning ``SquarePartitioner(num_partitions)``
* Exponential partitioning ``ExponentialPartitioner(num_partitions)``
* more to come in the future releases (contributions are welcome).

.. raw:: html

    <p align="center">
      <img src="./doc/source/_static/readme/comparison_of_partitioning_schemes.png" alt="Comparison of partitioning schemes."/>
      <br>
      <em>Comparison of Partitioning Schemes on CIFAR10</em>
    </p>

PS: This plot was generated using a library function (see `flwr_datasets.visualization <https://flower.ai/docs/datasets/ref-api/flwr_datasets.visualization.html>`_ package for more).

Usage
-----

Flower Datasets exposes the ``FederatedDataset`` abstraction to represent the dataset needed for federated learning/evaluation/analytics. It has two powerful methods that let you handle the dataset preprocessing: ``load_partition(partition_id, split)`` and ``load_split(split)``.

Here's a basic quickstart example of how to partition the MNIST dataset:

.. code-block:: python

    from flwr_datasets import FederatedDataset

    # The train split of the MNIST dataset will be partitioned into 100 partitions
    fds = FederatedDataset("mnist", partitioners={"train": 100})

    partition = fds.load_partition(0)

    centralized_data = fds.load_split("test")

For more details, please refer to the specific how-to guides or tutorial. They showcase customization and more advanced features.

Future release
--------------

Here are a few of the things that we will work on in future releases:

* ✅ Support for more datasets (especially the ones that have user id present).
* ✅ Creation of custom ``Partitioner``\ s.
* ✅ More out-of-the-box ``Partitioner``\ s.
* ✅ Passing ``Partitioner``\ s via ``FederatedDataset``'s ``partitioners`` argument.
* ✅ Customization of the dataset splitting before the partitioning.
* ✅ Simplification of the dataset transformation to the popular frameworks/types.
* Creation of the synthetic data,
* Support for Vertical FL.
