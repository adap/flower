:og:description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.

.. _quickstart-pytorch:

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.common.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.common.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.FedAvg.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.Strategy.html#flwr.serverapp.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.Strategy.html

Quickstart scikit-learn
=======================

In this federated learning tutorial we will learn how to train a Logistic Regression on
MNIST using Flower and scikit-learn. It is recommended to create a virtual environment
and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+scikit-learn project. It will
generate all the files needed to run, by default with the Flower Simulation Engine, a
federation of 10 nodes using |fedavg|_ The dataset will be partitioned using
|flowerdatasets|_'s |iidpartitioner|_

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

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

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration... 
    Success
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (0.06 MB)
    INFO :          ├── ConfigRecord (train): (empty!)
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (1.00) | evaluate ( 1.00)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'
    INFO :      
    INFO :      
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_logloss': 1.3937176081476854}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 1.2330690008199938, 'accuracy': 0.6915833333333334, 'precision': 0.686590664385589, 'recall': 0.6804619911253561, 'f1': 0.6575157112942838}
    INFO :      
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_logloss': 0.8565170774432291}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 0.8805567523494777, 'accuracy': 0.7342500000000001, 'precision': 0.7923715440451836, 'recall': 0.7329471009556615, 'f1': 0.7043857103531533}
    INFO :      
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_logloss': 0.703260769576}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 0.7020750690299385, 'accuracy': 0.7725000000000002, 'precision': 0.8220079490221321, 'recall': 0.7634786024767164, 'f1': 0.7506870949907579}
    INFO :      
    INFO :      Strategy execution finished in 17.87s
    INFO :      
    INFO :      Final results:
    INFO :      
    INFO :          Global Arrays:
    INFO :                  ArrayRecord (0.060 MB)
    INFO :      
    INFO :          Aggregated ClientApp-side Train Metrics:
    INFO :          { 1: {'train_logloss': '1.3937e+00'},
    INFO :            2: {'train_logloss': '8.5652e-01'},
    INFO :            3: {'train_logloss': '7.0326e-01'}}
    INFO :      
    INFO :          Aggregated ClientApp-side Evaluate Metrics:
    INFO :          { 1: { 'accuracy': '6.9158e-01',
    INFO :                 'f1': '6.5752e-01',
    INFO :                 'precision': '6.8659e-01',
    INFO :                 'recall': '6.8046e-01',
    INFO :                 'test_logloss': '1.2331e+00'},
    INFO :            2: { 'accuracy': '7.3425e-01',
    INFO :                 'f1': '7.0439e-01',
    INFO :                 'precision': '7.9237e-01',
    INFO :                 'recall': '7.3295e-01',
    INFO :                 'test_logloss': '8.8056e-01'},
    INFO :            3: { 'accuracy': '7.7250e-01',
    INFO :                 'f1': '7.5069e-01',
    INFO :                 'precision': '8.2201e-01',
    INFO :                 'recall': '7.6348e-01',
    INFO :                 'test_logloss': '7.0208e-01'}}
    INFO :      
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    INFO :      

    Saving final model to disk...

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
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Create LogisticRegression Model
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)
        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(model)

        # Apply received pararameters
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        set_model_params(model, ndarrays)

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        X_train, _, y_train, _ = load_data(partition_id, num_partitions)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Train the model on local data
            model.fit(X_train, y_train)

        # Let's compute train loss
        y_train_pred_proba = model.predict_proba(X_train)
        train_logloss = log_loss(y_train, y_train_pred_proba)

        # Construct and return reply Message
        ndarrays = get_model_params(model)
        model_record = ArrayRecord(ndarrays)
        metrics = {"num-examples": len(X_train), "train_logloss": train_logloss}
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

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
        """Main entry point for the ServerApp."""

        # Read run config
        num_rounds: int = context.run_config["num-server-rounds"]

        # Create LogisticRegression Model
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)
        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(model)
        # Construct ArrayRecord representation
        arrays = ArrayRecord(get_model_params(model))

        # Initialize FedAvg strategy
        strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model parameters
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        set_model_params(model, ndarrays)
        joblib.dump(model, "logreg_model.pkl")

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

.. |quickstart_sklearn_link| replace:: ``examples/sklearn-logreg-mnist``

.. _client: ref-api/flwr.client.Client.html#client

.. _fedavg: ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg

.. _flowerdatasets: https://flower.ai/docs/datasets/

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner

.. _logisticregression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

.. meta::
    :description: Check out this Federated Learning quickstart tutorial for using Flower with scikit-learn to train a linear regression model.
