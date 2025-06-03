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
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
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
    INFO :      Run finished 3 round(s) in 19.41s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 1.3447584261018466
    INFO :                  round 2: 0.9680018613482815
    INFO :                  round 3: 0.7667920399137523
    INFO :

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

To perform the training and evaluation, we will make use of the ``.fit()`` and
``.score()`` methods available in the ``LogisticRegression`` class.

The ClientApp
-------------

The main changes we have to make to use scikit-learn with Flower will be found in the
``get_model_params()``, ``set_model_params()``, and ``set_initial_params()`` functions.
In ``get_model_params()``, the coefficients and intercept of the logistic regression
model are extracted and represented as a list of NumPy arrays. In
``set_model_params()``, that's the opposite: given a list of NumPy arrays it applies
them to an existing ``LogisticRegression`` model. Finally, in ``set_initial_params()``,
we initialize the model parameters based on the MNIST dataset, which has 10 classes
(corresponding to the 10 digits) and 784 features (corresponding to the size of the
MNIST image array, which is 28 × 28). Doing this is fairly easy in scikit-learn.

.. code-block:: python

    def get_model_params(model):
        if model.fit_intercept:
            params = [
                model.coef_,
                model.intercept_,
            ]
        else:
            params = [model.coef_]
        return params


    def set_model_params(model, params):
        model.coef_ = params[0]
        if model.fit_intercept:
            model.intercept_ = params[1]
        return model


    def set_initial_params(model):
        n_classes = 10  # MNIST has 10 classes
        n_features = 784  # Number of features in dataset
        model.classes_ = np.array([i for i in range(10)])

        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))

The rest of the functionality is directly inspired by the centralized case:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, model, X_train, X_test, y_train, y_test):
            self.model = model
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

        def fit(self, parameters, config):
            set_model_params(self.model, parameters)

            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.X_train, self.y_train)

            return get_model_params(self.model), len(self.X_train), {}

        def evaluate(self, parameters, config):
            set_model_params(self.model, parameters)
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            accuracy = self.model.score(self.X_test, self.y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}

Finally, we can construct a ``ClientApp`` using the ``FlowerClient`` defined above by
means of a ``client_fn()`` callback. Note that the ``context`` enables you to get access
to hyperparameters defined in your ``pyproject.toml`` to configure the run. In this
tutorial we access the `local-epochs` setting to control the number of epochs a
``ClientApp`` will perform when running the ``fit()`` method. You could define
additional hyperparameters in ``pyproject.toml`` and access them here.

.. code-block:: python

    def client_fn(context: Context):
        # Load data and model
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)
        penalty = context.run_config["penalty"]
        local_epochs = context.run_config["local-epochs"]
        model = get_model(penalty, local_epochs)

        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(model)

        # Return Client instance
        return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn)

The ServerApp
-------------

To construct a ``ServerApp`` we define a ``server_fn()`` callback with an identical
signature to that of ``client_fn()`` but the return type is |serverappcomponents|_ as
opposed to a |client|_ In this example we use the `FedAvg` strategy. To it we pass a
zero-initialized model that will server as the global model to be federated. Note that
the values of ``num-server-rounds``, ``penalty``, and ``local-epochs`` are read from the
run config. You can find the default values defined in the ``pyproject.toml``.

.. code-block:: python

    def server_fn(context: Context):
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

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Congratulations! You've successfully built and run your first federated learning system
in scikit-learn.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_sklearn_link|_ in the Flower GitHub repository.

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

.. meta::
    :description: Check out this Federated Learning quickstart tutorial for using Flower with scikit-learn to train a linear regression model.
