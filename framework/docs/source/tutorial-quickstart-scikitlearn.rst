:og:description: Learn how to train a logistic regression on the Iris dataset using federated learning with Flower and scikit-learn in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a logistic regression on the Iris dataset using federated learning with Flower and scikit-learn in this step-by-step tutorial.

.. _quickstart-sklearn-tabular:

Quickstart scikit-learn (tabular)
=================================

In this federated learning tutorial we will learn how to train a Logistic Regression on
the Iris dataset using Flower and scikit-learn. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

We'll use ``flwr new`` to create a complete Flower+scikit-learn project scaffold. It
will generate all the files needed to run, by default with the Flower Simulation Engine,
a federation of 10 nodes using |fedavg|_. Instead of MNIST, we will work with the
classic Iris dataset.

First, install Flower in your environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. Choose the ``sklearn`` template, provide a project name,
and your developer name:

.. code-block:: shell

    $ flwr new

After running it you'll see a new directory with this structure:

.. code-block:: shell

    <your-project-name>
    ├── <your-project-name>
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp (@app.train / @app.evaluate)
    │   ├── server_app.py   # Defines your ServerApp (@app.main)
    │   └── task.py         # Defines model, training, evaluation, and data loading
    ├── pyproject.toml
    └── README.md

Install the project and dependencies:

.. code-block:: shell

    $ pip install -e .

Run the project:

.. code-block:: shell

    $ flwr run .

With default arguments you should see an output like this:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :          ├── ArrayRecord (small, ~kB)
    INFO :          ├── ConfigRecord (train): {'lr': 0.1}
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (1.00) | evaluate (0.50)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'

You can override config values defined in ``[tool.flwr.app.config]`` in
``pyproject.toml``:

.. code-block:: shell

    $ flwr run . --run-config "num-server-rounds=5 local-epochs=2"

The Data
--------

This tutorial uses scikit-learn’s built-in Iris dataset. We split it into 10 partitions
(one for each client) and within each partition, 80% is used for training and 20% for
testing:

.. code-block:: python

    iris = load_iris()
    X, y = iris.data, iris.target

    # Partition data
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

The Model
---------

We define the |logisticregression|_ model in ``task.py``:

.. code-block:: python

    def get_model(penalty: str, local_epochs: int):
        return LogisticRegression(
            penalty=penalty,
            max_iter=local_epochs,
            warm_start=True,
        )

The ClientApp
-------------

Instead of subclassing ``NumPyClient``, the new API uses decorators. In
``client_app.py`` you’ll see:

.. code-block:: python

    app = ClientApp()


    @app.train()
    def train(config: Config, data: Tuple) -> RecordDict:
        # set model params, train locally, return updated weights + metrics
        ...


    @app.evaluate()
    def evaluate(config: Config, data: Tuple) -> RecordDict:
        # set model params, evaluate locally, return loss + accuracy
        ...

The ServerApp
-------------

In ``server_app.py`` the federated averaging strategy is defined:

.. code-block:: python

    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        initial_arrays = ArrayRecord.from_numpy_ndarrays(get_model_params(model))
        strategy = FedAvg(
            min_available_nodes=2,
            train_metrics_aggr_fn=weighted_average,
            evaluate_metrics_aggr_fn=weighted_average,
        )
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=context.run_config["num-server-rounds"],
        )

Congratulations! You've now built and run your first federated learning system in
scikit-learn on the Iris dataset using the new Message API.

.. note::

    Check the source code of this tutorial in the `Flower GitHub repository
    <https://github.com/adap/flower/tree/main/examples/quickstart-sklearn-tabular>`_.

.. |fedavg| replace:: ``FedAvg``

.. |logisticregression| replace:: ``LogisticRegression``
