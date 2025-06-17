:og:description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.

.. _quickstart-fastai:

Quickstart fastai
=================

In this federated learning tutorial we will learn how to train a SqueezeNet model on
MNIST using Flower and fastai. It is recommended to create a virtual environment and run
everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Then, clone the code example directly from GitHub:

.. code-block:: shell

    git clone --depth=1 https://github.com/adap/flower.git _tmp \
                 && mv _tmp/examples/quickstart-fastai . \
                 && rm -rf _tmp && cd quickstart-fastai

This will create a new directory called `quickstart-fastai` containing the following
files:

.. code-block:: shell

    quickstart-fastai
    ├── fastai_example
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, activate your environment, then run:

.. code-block:: shell

    # Navigate to the example directory
    $ cd path/to/quickstart-fastai

    # Install project and dependencies
    $ pip install -e .

This example by default runs the Flower Simulation Engine, creating a federation of 10
nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_
as the aggregation strategy. The dataset will be partitioned using Flower Dataset's
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.
Let's run the project:

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
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_evaluate: received 5 results and 0 failures
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_evaluate: received 5 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_evaluate: received 5 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 143.02s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 2.699497365951538
    INFO :                  round 2: 0.9549586296081543
    INFO :                  round 3: 0.6627192616462707
    INFO :          History (metrics, distributed, evaluate):
    INFO :          {'accuracy': [(1, 0.09766666889190674),
    INFO :                        (2, 0.6948333323001862),
    INFO :                        (3, 0.7721666693687439)]}
    INFO :

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config num-server-rounds=5

.. note::

    Check the `source code
    <https://github.com/adap/flower/tree/main/examples/quickstart-fastai>`_ of this
    tutorial in ``examples/quickstart-fastai`` in the Flower GitHub repository.
