:og:description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a logistic regression on MNIST using federated learning with Flower and scikit-learn in this step-by-step tutorial.

.. _quickstart-pytorch:

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

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
federation of 10 nodes using |fedavg_link|_ The dataset will be partitioned using
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
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 1.23306, 'accuracy': 0.69154, 'precision': 0.68659, 'recall': 0.68046, 'f1': 0.65752}
    INFO :
    INFO :      [ROUND 2/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_logloss': 0.8565170774432291}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 0.8805, 'accuracy': 0.73425, 'precision': 0.792371, 'recall': 0.7329, 'f1': 0.70438}
    INFO :
    INFO :      [ROUND 3/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_logloss': 0.703260769576}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'test_logloss': 0.70207, 'accuracy': 0.77250, 'precision': 0.82201, 'recall': 0.76348, 'f1': 0.75069}
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

The main changes we have to make to use ``Scikit-learn`` with ``Flower`` have to do with
converting the |arrayrecord_link|_ received in the |message_link|_ into numpy ndarrays
and then use them to set the model parameters. After training, another auxiliary
function can be used to extract then pack the updated numpy ndarrays into a ``Message``
from the ClientApp. We can make use of built-in methods in the ``ArrayRecord`` to make
these conversions:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):

        # Load the model
        model = get_model()  # construct your scikit-learn model
        # Extract the ArrayRecord from Message and convert to numpy ndarrays
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        # Set the model parameters with auxhiliary function
        set_model_params(model, ndarrays)

        # Train the model
        ...

        # Extract the updated model parameters with auxhiliary function
        updated_ndarrays = get_model_params(model)
        # Pack the updated parameters into an ArrayRecord
        model_record = ArrayRecord(updated_ndarrays)

The rest of the functionality is directly inspired by the centralized case. The
|clientapp_link|_ comes with three core methods (``train``, ``evaluate``, and ``query``)
that we can implement for different purposes. For example: ``train`` to train the
received model using the local data; ``evaluate`` to assess its performance of the
received model on a validation set; and ``query`` to retrieve information about the node
executing the ``ClientApp``. In this tutorial we will only make use of ``train`` and
``evaluate``.

Let's see how the ``train`` method can be implemented. It receives as input arguments a
|message_link|_ from the ``ServerApp``. By default it carries:

- an ``ArrayRecord`` with the arrays of the model to federate. By default they can be
  retrieved with key ``"arrays"`` when accessing the message content.
- a ``ConfigRecord`` with the configuration sent from the ``ServerApp``. By default it
  can be retrieved with key ``"config"`` when accessing the message content.

The ``train`` method also receives the ``Context``, giving access to configs for your
run and node. The run config hyperparameters are defined in the ``pyproject.toml`` of
your Flower App. The node config can only be set when running Flower with the Deployment
Runtime and is not directly configurable during simulations.

.. code-block:: python

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

To construct a |serverapp_link|_ we define its ``@app.main()`` method. This method
receive as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the |fedavg_link|_ and configure it with a specific value of
``fraction_train`` which is read from the run config. You can find the default value
defined in the ``pyproject.toml``. Then, the execution of the strategy is launched when
invoking its |strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to federated.
- a ``ConfigRecord`` with the training hyperparameters to be sent to the clients. The
  strategy will also insert the current round number in this config before sending it to
  the participating nodes.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

.. code-block:: python

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

    Check the source code of another Flower App using ``scikit-learn`` in the `Flower
    GitHub repository
    <https://github.com/adap/flower/tree/main/examples/quickstart-sklearn-tabular>`_.

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
