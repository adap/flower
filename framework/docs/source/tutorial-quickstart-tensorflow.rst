:og:description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and TensorFlow in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and TensorFlow in this step-by-step tutorial.

.. _quickstart-tensorflow:

Quickstart TensorFlow
=====================

In this tutorial we will learn how to train a Convolutional Neural Network on CIFAR-10
using the Flower framework and TensorFlow. First of all, it is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use `flwr new` to create a complete Flower+TensorFlow project. It will generate
all the files needed to run, by default with the Flower Simulation Engine, a federation
of 10 nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_.
The dataset will be partitioned using Flower Dataset's `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. You will be prompted to select one of the available
templates (choose ``TensorFlow``), give a name to your project, and type in your
developer name:

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
    INFO :      Run finished 3 round(s) in 31.31s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 1.9066195368766785
    INFO :                  round 2: 1.657227087020874
    INFO :                  round 3: 1.559039831161499
    INFO :

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 batch-size=16"

The Data
--------

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily
download and partition the `CIFAR-10` dataset. In this example you'll make use of the
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose `other partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets. Each ``ClientApp`` will call this function to create the ``NumPy``
arrays that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

The Model
---------

Next, we need a model. We defined a simple Convolutional Neural Network (CNN), but feel
free to replace it with a more sophisticated model if you'd like:

.. code-block:: python

    def load_model(learning_rate: float = 0.001):
        # Define a simple CNN for CIFAR-10 and set Adam optimizer
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            "adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

The ClientApp
-------------

With `TensorFlow`, we can use the built-in ``get_weights()`` and ``set_weights()``
functions, which simplifies the implementation with `Flower`. The rest of the
functionality in the ClientApp is directly inspired by the centralized case. The
``fit()`` method in the client trains the model using the local dataset. Similarly, the
``evaluate()`` method is used to evaluate the model received on a held-out validation
set that the client might have:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, model, data, epochs, batch_size, verbose):
            self.model = model
            self.x_train, self.y_train, self.x_test, self.y_test = data
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose

        def fit(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            return self.model.get_weights(), len(self.x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            return loss, len(self.x_test), {"accuracy": accuracy}

Finally, we can construct a ``ClientApp`` using the ``FlowerClient`` defined above by
means of a ``client_fn()`` callback. Note that the `context` enables you to get access
to hyperparameters defined in your ``pyproject.toml`` to configure the run. For example,
in this tutorial we access the `local-epochs` setting to control the number of epochs a
``ClientApp`` will perform when running the ``fit()`` method, in addition to
`batch-size`. You could define additional hyperparameters in ``pyproject.toml`` and
access them here.

.. code-block:: python

    def client_fn(context: Context):
        # Load model and data
        net = load_model()

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions)
        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose")

        # Return Client instance
        return FlowerClient(net, data, epochs, batch_size, verbose).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn=client_fn)

The ServerApp
-------------

To construct a ``ServerApp`` we define a ``server_fn()`` callback with an identical
signature to that of ``client_fn()`` but the return type is `ServerAppComponents
<https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html#serverappcomponents>`_
as opposed to a `Client
<https://flower.ai/docs/framework/ref-api/flwr.client.Client.html#client>`_. In this
example we use the `FedAvg`. To it we pass a randomly initialized model that will serve
as the global model to federate.

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Get parameters to initialize global model
        parameters = ndarrays_to_parameters(load_model().get_weights())

        # Define strategy
        strategy = strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Congratulations! You've successfully built and run your first federated learning system.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_tf_link|_ in the Flower GitHub repository.

.. |quickstart_tf_link| replace:: ``examples/quickstart-tensorflow``

.. _quickstart_tf_link: https://github.com/adap/flower/blob/main/examples/quickstart-tensorflow
