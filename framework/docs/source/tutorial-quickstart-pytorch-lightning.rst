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
4 nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_
as the aggregation strategy. The dataset will be partitioned using Flower Dataset's
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.
To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Using initial global parameters provided by strategy
    INFO :      Starting evaluation of initial global parameters
    INFO :      Evaluation returned no results (`None`)
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 4)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 2 clients (out of 4)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 4)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 2 clients (out of 4)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 4)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 136.92s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 0.04982871934771538
    INFO :                  round 2: 0.046457378193736076
    INFO :                  round 3: 0.04506748169660568
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
