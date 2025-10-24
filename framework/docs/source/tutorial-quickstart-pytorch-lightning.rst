:og:description: Learn how to train an autoencoder on MNIST using federated learning with Flower and PyTorch Lightning in this step-by-step tutorial.
.. meta::
    :description: Learn how to train an autoencoder on MNIST using federated learning with Flower and PyTorch Lightning in this step-by-step tutorial.

.. _quickstart-pytorch-lightning:

Quickstart PyTorch Lightning
============================

In this federated learning tutorial we will learn how to train an AutoEncoder model on
MNIST using Flower and PyTorch Lightning. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Then, clone the code example directly from GitHub:

.. code-block:: shell

    git clone --depth=1 https://github.com/adap/flower.git _tmp \
                 && mv _tmp/examples/quickstart-pytorch-lightning . \
                 && rm -rf _tmp && cd quickstart-pytorch-lightning

This will create a new directory called `quickstart-pytorch-lightning` containing the
following files:

.. code-block:: shell

    quickstart-pytorch-lightning
    ├── pytorchlightning_example
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, activate your environment, then run:

.. code-block:: shell

    # Navigate to the example directory
    $ cd path/to/quickstart-pytorch-lightning

    # Install project and dependencies
    $ pip install -e .

By default, Flower Simulation Engine will be started and it will create a federation of
4 nodes using |fedavg|_ as the aggregation strategy. The dataset will be partitioned
using Flower Dataset's |iidpartitioner|_. To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (0.39 MB)
    INFO :          ├── ConfigRecord (train): (empty!)
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (0.50) | evaluate ( 0.50)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'
    INFO :
    INFO :
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 2 nodes (out of 4)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.0487}
    INFO :      configure_evaluate: Sampled 2 nodes (out of 4)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.0495}
    INFO :
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 2 nodes (out of 4)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.0420}
    INFO :      configure_evaluate: Sampled 2 nodes (out of 4)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.0455}
    INFO :
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 2 nodes (out of 4)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.05082}
    INFO :      configure_evaluate: Sampled 2 nodes (out of 4)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.0441}
    INFO :
    INFO :      Strategy execution finished in 159.24s
    INFO :
    INFO :      Final results:
    INFO :
    INFO :          Global Arrays:
    INFO :                  ArrayRecord (0.389 MB)
    INFO :
    INFO :          Aggregated ClientApp-side Train Metrics:
    INFO :          { 1: {'train_loss': '4.8696e-02'},
    INFO :            2: {'train_loss': '4.1957e-02'},
    INFO :            3: {'train_loss': '5.0818e-02'}}
    INFO :
    INFO :          Aggregated ClientApp-side Evaluate Metrics:
    INFO :          { 1: {'eval_loss': '4.9516e-02'},
    INFO :            2: {'eval_loss': '4.5510e-02'},
    INFO :            3: {'eval_loss': '4.4052e-02'}}
    INFO :
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    INFO :

Each simulated `ClientApp` (two per round) will also log a summary of their local
training process. Expect this output to be similar to:

.. code-block:: shell

    # The left part indicates the process ID running the `ClientApp`
    (ClientAppActor pid=38155) ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    (ClientAppActor pid=38155) ┃        Test metric        ┃       DataLoader 0        ┃
    (ClientAppActor pid=38155) ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    (ClientAppActor pid=38155) │         test_loss         │   0.045175597071647644    │
    (ClientAppActor pid=38155) └───────────────────────────┴───────────────────────────┘

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config num-server-rounds=5

.. note::

    Check the `source code
    <https://github.com/adap/flower/tree/main/examples/quickstart-pytorch-lightning>`_
    of this tutorial in ``examples/quickstart-pytorch-lightning`` in the Flower GitHub
    repository.

.. |fedavg| replace:: ``FedAvg``

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |iidpartitioner| replace:: ``IidPartitioner``

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner
