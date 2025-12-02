:og:description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.

.. _quickstart-fastai:

###################
 Quickstart fastai
###################

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
<https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_
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
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (4.72 MB)
    INFO :          ├── ConfigRecord (train): (empty!)
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (0.50) | evaluate ( 1.00)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'
    INFO :
    INFO :
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 3.1197, 'eval_acc': 0.14874}
    INFO :
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.8071, 'eval_acc': 0.7488}
    INFO :
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.5015, 'eval_acc': 0.8547}
    INFO :
    INFO :      Strategy execution finished in 72.84s
    INFO :
    INFO :      Final results:
    INFO :
    INFO :          Global Arrays:
    INFO :                  ArrayRecord (4.719 MB)
    INFO :
    INFO :          Aggregated ClientApp-side Train Metrics:
    INFO :          {1: {}, 2: {}, 3: {}}
    INFO :
    INFO :          Aggregated ClientApp-side Evaluate Metrics:
    INFO :          { 1: {'eval_acc': '1.4875e-01', 'eval_loss': '3.1197e+00'},
    INFO :            2: {'eval_acc': '7.4883e-01', 'eval_loss': '8.0705e-01'},
    INFO :            3: {'eval_acc': '8.5467e-01', 'eval_loss': '5.0145e-01'}}
    INFO :
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    INFO :

    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config num-server-rounds=5

.. note::

    Check the `source code
    <https://github.com/adap/flower/tree/main/examples/quickstart-fastai>`_ of this
    tutorial in ``examples/quickstart-fastai`` in the Flower GitHub repository.
