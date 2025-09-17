:og:description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.

.. _quickstart-scikitlearn:

Quickstart scikit-learn
=======================

In this federated learning tutorial we will learn how to train a Logistic Regression on
the MNIST dataset using Flower and scikit-learn. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

We'll use ``flwr new`` to create a complete Flower+scikit-learn project scaffold. It
will generate all the files needed to run, by default with the Flower Simulation Engine,
a federation of 10 nodes using |fedavg|_. The dataset will be partitioned using
|flowerdatasets|_'s |iidpartitioner|_.

First, install Flower in your environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. You will be prompted to select one of the available
templates (choose ``sklearn``), give a name to your project, and type in your developer
name:

.. code-block:: shell

    $ flwr new

After running it you'll notice a new directory with your project name has been created.
It should have the following structure:

.. code-block:: shell

    <your-project-name>
    ├── <your-project-name>
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

If you haven't yet installed the project and its dependencies, you can do so by:

.. code-block:: shell

    # From the directory where your pyproject.toml is
    $ pip install -e .

To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you should see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (0.24 MB)
    INFO :          ├── ConfigRecord (train): {'penalty': 'l2'}
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (1.00) | evaluate (0.50)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'
    INFO :
    INFO :      Initial global evaluation results: {'accuracy': 0.1, 'loss': 2.30}
    INFO :
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.10}
    INFO :      configure_evaluate: Sampled 5 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.20, 'eval_accuracy': 0.12}
    INFO :      Global evaluation
    INFO :          └──> MetricRecord: {'accuracy': 0.11, 'loss': 2.22}
    INFO :
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.00}
    INFO :      configure_evaluate: Sampled 5 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.05, 'eval_accuracy': 0.20}
    INFO :      Global evaluation
    INFO :          └──> MetricRecord: {'accuracy': 0.19, 'loss': 2.05}
    INFO :
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 1.95}
    INFO :      configure_evaluate: Sampled 5 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 1.90, 'eval_accuracy': 0.28}
    INFO :      Global evaluation
    INFO :          └──> MetricRecord: {'accuracy': 0.25, 'loss': 1.95}
    INFO :
    INFO :      Strategy execution finished in XX.XXs
    INFO :
    INFO :      Final results:
    INFO :
    INFO :          Global Arrays:
    INFO :                  ArrayRecord (0.23 MB)
    INFO :
    INFO :          Aggregated ClientApp-side Train Metrics:
    INFO :          { 1: {'train_loss': '2.10e+00'},
    INFO :            2: {'train_loss': '2.00e+00'},
    INFO :            3: {'train_loss': '1.95e+00'}}
    INFO :
    INFO :          Aggregated ClientApp-side Evaluate Metrics:
    INFO :          { 1: {'eval_accuracy': '1.20e-01', 'eval_loss': '2.20e+00'},
    INFO :            2: {'eval_accuracy': '2.00e-01', 'eval_loss': '2.05e+00'},
    INFO :            3: {'eval_accuracy': '2.80e-01', 'eval_loss': '1.90e+00'}}
    INFO :
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          { 0: {'accuracy': '1.00e-01', 'loss': '2.30e+00'},
    INFO :            1: {'accuracy': '1.10e-01', 'loss': '2.22e+00'},
    INFO :            2: {'accuracy': '1.90e-01', 'loss': '2.05e+00'},
    INFO :            3: {'accuracy': '2.50e-01', 'loss': '1.95e+00'}}

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 local-epochs=2"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

The Data
--------

This tutorial uses |flowerdatasets|_ to easily download and partition the `MNIST
<https://huggingface.co/datasets/ylecun/mnist>`_ dataset. In this example you'll make
use of the |iidpartitioner|_ to generate ``num_partitions`` partitions. You can choose
|otherpartitioners|_ available in Flower Datasets. Each ``ClientApp`` will call this
function to create dataloaders with the data that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": partitioner},
    )

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")

    X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

The Model
---------

We define the |logisticregression|_ model from scikit-learn in the ``get_model()``
function:

.. code-block:: python

    def get_model(penalty: str, local_epochs: int):
        return LogisticRegression(
            penalty=penalty,
            max_iter=local_epochs,
            warm_start=True,
        )

The ClientApp
-------------

The new Message API defines clients via the ``ClientApp`` class and decorators. Each
client implements two functions—\ ``train`` and ``evaluate``\ — which operate on a
``Message`` and return a ``Message``. A ``Message`` received from the server carries the
current global model weights as an ``ArrayRecord`` (stored under the key ``"arrays"``)
and an optional ``ConfigRecord`` with hyperparameters (stored under the key
``"config"``). The ``Context`` parameter gives access to the run configuration defined
in your ``pyproject.toml`` and, when running on the Deployment Engine, the node
configuration. In this example we only use the run configuration to read the penalty and
number of local epochs.

A typical ``train`` method for logistic regression looks like this:

.. code-block:: python

    from flwr.app import ArrayRecord, MetricRecord, RecordDict, Message
    from typing import Tuple

    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context) -> Message:
        """Handle a training request from the server."""
        # 1) Instantiate a logistic regression model and
        # set its parameters from the received ArrayRecord.
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        model = set_model_params(model, ndarrays)

        # 2) Load the local training data.
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        X_train, _, y_train, _ = load_data(partition_id, num_partitions)

        # 3) Fit the model on the local data.
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)

        # 4) Build the reply Message.
        arrays_record = ArrayRecord.from_numpy_ndarrays(get_model_params(model))
        metrics = MetricRecord(
            {
                "train_accuracy": train_accuracy,
                "num-examples": len(X_train),
            }
        )
        reply_content = RecordDict({"arrays": arrays_record, "metrics": metrics})
        return Message(content=reply_content, reply_to=msg)

The ``@app.evaluate`` method mirrors ``train`` but only evaluates the received model on
the local validation set. It returns a ``MetricRecord`` containing the evaluation loss
and accuracy and does not include the model weights, since they are not modified during
evaluation.

The ServerApp
-------------

The server runs a ``ServerApp`` which contains a single entrypoint annotated with
``@app.main()``. This function receives two arguments:

- **grid** – an instance of ``Grid`` used to communicate with the participating nodes
  running the ``ClientApp``. It abstracts details of the underlying transport (e.g.,
  gRPC, HTTP) and allows the ``ServerApp`` to broadcast requests and gather replies.
- **context** – a ``Context`` providing access to the run configuration. From here you
  can read values defined in your ``pyproject.toml``, such as the number of server
  rounds, the regularisation penalty for logistic regression, or the number of local
  epochs to be performed on each client.

Within the ``main`` method you typically:

1. **Create the global model** and wrap its parameters in an ``ArrayRecord``. For
   scikit-learn we instantiate a ``LogisticRegression`` model with the desired penalty
   and maximum number of iterations and convert its coefficients and intercept into a
   list of NumPy arrays via ``get_model_params``.
2. **Initialize the strategy**. In this tutorial we use |fedavg|_ with two custom
   aggregation functions: ``train_metrics_aggr_fn`` and ``evaluate_metrics_aggr_fn``.
   These functions compute a weighted average of client metrics using the number of
   examples processed on each client as the weight. Passing them to the strategy ensures
   that ``train_loss`` and ``eval_accuracy`` are aggregated correctly across clients.
3. **Launch the federated training loop** by calling ``strategy.start``. You must pass
   the ``grid``, the ``initial_arrays`` (the model parameters), and ``num_rounds``
   specifying how many rounds of `FedAvg` to perform.

Here is a simplified version of the ``main`` method:

.. code-block:: python

    from flwr.app import ArrayRecord
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg

    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Entry point for the server."""
        # 1) Build the initial logistic regression model
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)
        initial_arrays = ArrayRecord.from_numpy_ndarrays(get_model_params(model))

        # 2) Configure the strategy.  Use the weighted average functions
        # to aggregate client-side metrics.
        min_available_nodes = context.run_config["min-available-clients"]
        strategy = FedAvg(
            min_available_nodes=min_available_nodes,
            train_metrics_aggr_fn=weighted_average,
            evaluate_metrics_aggr_fn=weighted_average,
        )

        # 3) Start federated learning.  Run FedAvg for the specified number of rounds.
        num_rounds = context.run_config["num-server-rounds"]
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
        )

        # 4) Print or save the final model and metrics (optional)
        print(result)

Congratulations! You've successfully built and run your first federated learning system
in scikit-learn on the MNIST dataset using the new Message API.

.. note::

    Check the source code of this tutorial in the `Flower GitHub repository
    <https://github.com/adap/flower/tree/main/examples/quickstart-sklearn-tabular>`_.

.. |client| replace:: ``Client``

.. |fedavg| replace:: ``FedAvg``

.. |flowerdatasets| replace:: Flower Datasets

.. |iidpartitioner| replace:: ``IidPartitioner``

.. |logisticregression| replace:: ``LogisticRegression``

.. |otherpartitioners| replace:: other partitioners

.. |serverappcomponents| replace:: ``ServerAppComponents``

.. |quickstart_sklearn_link| replace:: ``examples/sklearn-logreg-mnist``

.. _client: ref-api/flwr.client.Client.html#client

.. _fedavg: ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg

.. _flowerdatasets: https://flower.ai/docs/datasets/

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner

.. _logisticregression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

.. _quickstart_sklearn_link: https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist

.. _serverappcomponents: ref-api/flwr.server.ServerAppComponents.html#serverappcomponents
