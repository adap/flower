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
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Using initial global parameters provided by strategy
    INFO :      Starting evaluation of initial global parameters
    INFO :      Evaluation returned no results (`None`)
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 14.53s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 1.233069000819992
    INFO :                  round 2: 0.8805567523494775
    INFO :                  round 3: 0.7020750690299342

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
        # Instantiate a logistic regression model
        n_classes = 10  # MNIST has 10 classes
        n_features = 784  # Number of features in dataset
        model.classes_ = np.array([i for i in range(10)])

        # ...

        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))

        # 2) Fit the model
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

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
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Create LogisticRegression Model
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)

        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(model)

        initial_parameters = ndarrays_to_parameters(get_model_params(model))

        # Define strategy
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=initial_parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

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
