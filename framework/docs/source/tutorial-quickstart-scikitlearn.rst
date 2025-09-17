:og:description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.

.. _quickstart-scikitlearn:

Quickstart scikit-learn
=======================

In this federated learning tutorial we will learn how to train a Logistic Regression on
MNIST using Flower and scikit-learn. It is recommended to create a virtual environment
and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+scikit-learn project. It will
generate all the files needed to run, by default with the Flower Simulation Engine, a
federation of 10 nodes using |fedavg|_. The dataset will be partitioned using
|flowerdatasets|_'s |iidpartitioner|_.

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
    │   ├── client_app.py   # Defines your ClientApp (@app.train / @app.evaluate)
    │   ├── server_app.py   # Defines your ServerApp (@app.main)
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
    INFO :          ├── ArrayRecord (0.24 MB)
    INFO :          ├── ConfigRecord (train): {'lr': 0.1}
    INFO :          ├── ConfigRecord (evaluate): (empty!)
    INFO :          ├──> Sampling:
    INFO :          │       ├──Fraction: train (1.00) | evaluate (0.50)
    INFO :          │       ├──Minimum nodes: train (2) | evaluate (2)
    INFO :          │       └──Minimum available nodes: 2
    INFO :          └──> Keys in records:
    INFO :                  ├── Weighted by: 'num-examples'
    INFO :                  ├── ArrayRecord key: 'arrays'
    INFO :                  └── ConfigRecord key: 'config'

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
    y_train, y_test = y[: int(0.8 * len(y)) :]

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

Instead of subclassing ``NumPyClient``, the new API uses decorators. You implement
``@app.train`` and ``@app.evaluate`` to handle training and evaluation on each client.

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

This allows direct use of |arrayrecord|_ and |metricrecord|_ for exchanging model
parameters and metrics, making the API more consistent across frameworks.

The ServerApp
-------------

Instead of ``server_fn`` returning ``ServerAppComponents``, the new API defines
a single entrypoint using ``@app.main``:

.. code-block:: python

    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        # Create initial model parameters
        initial_arrays = ArrayRecord.from_numpy_ndarrays(get_model_params(model))

        # Define strategy with aggregation functions
        strategy = FedAvg(
            min_available_nodes=2,
            train_metrics_aggr_fn=weighted_average,
            evaluate_metrics_aggr_fn=weighted_average,
        )

        # Run training
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=context.run_config["num-server-rounds"],
        )

Congratulations! You've successfully built and run your first federated learning system
in scikit-learn using the new Message API.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_sklearn_link|_ in the Flower GitHub repository.

.. |fedavg| replace:: ``FedAvg``
.. |flowerdatasets| replace:: Flower Datasets
.. |iidpartitioner| replace:: ``IidPartitioner``
.. |logisticregression| replace:: ``LogisticRegression``
.. |otherpartitioners| replace:: other partitioners
.. |arrayrecord| replace:: ``ArrayRecord``
.. |metricrecord| replace:: ``MetricRecord``

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html
.. _flowerdatasets: https://flower.ai/docs/datasets/
.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html
.. _logisticregression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html
.. _arrayrecord: ref-api/flwr.app.ArrayRecord.html
.. _metricrecord: ref-api/flwr.app.MetricRecord.html
.. _quickstart_sklearn_link: https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist

.. meta::
    :description: Check out this Federated Learning quickstart tutorial for using Flower with scikit-learn to train a logistic regression model.
