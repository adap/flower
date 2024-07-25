.. _quickstart-mlx:


Quickstart MLX
==============


In this tutorial we will learn how to train simple MLP on MNIST using Flower and MLX.

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Let's use `flwr new` to create a complete Flower+MLX project. It will generate all the files needed to run, by default with the Simulation Engine, a federation of 10 nodes using `FedAvg <https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>_`. The dataset will be partitioned using Flower Dataset's `IidPartitioner <https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>_`.

Now that we have a rough idea of what this example is about, let's get started. We first need to create an MLX project. You can do this by running the command below. You will be prompted to give a name to your project as well as typing your developer name.:

.. code-block:: shell

  $ flwr new --framework MLX

After running it you'll notice a new directory with your project name has been created. It should have the following structure:

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

To run the project do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default argumnets you will see an output like this one:

.. code-block:: shell
    Loading project configuration...
    Success
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    WARNING :   FAB ID is not provided; the default ClientApp will be loaded.
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
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
    INFO :      Run finished 3 round(s) in 8.15s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 2.243802046775818
    INFO :                  round 2: 2.101812958717346
    INFO :                  round 3: 1.7419301986694335
    INFO :


You can also override the parameters defined in `[tool.flwr.app.config]` section in the `pyproject.toml` like this:

.. code-block:: shell
    # Override some arguments
    $ flwr run . --run-config num-server-rounds=5,lr=0.05


What follows is an explanation of each component in the project you just created: dataset partition, the model, defining the `ClientApp` and defining the `ServerApp`.

The Data
--------

We will use `flwr_datasets` to easily download and partition the `MNIST` dataset.
In this example you'll make use of the `IidPartitioner <https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_ to generate `num_partitions` partitions.
You can choose `other partitioners <https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in Flower Datasets:

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    partition_splits = partition.train_test_split(test_size=0.2, seed=42)

    partition_splits["train"].set_format("numpy")
    partition_splits["test"].set_format("numpy")

    train_partition = partition_splits["train"].map(
        lambda img: {
            "img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0
        },
        input_columns="image",
    )
    test_partition = partition_splits["test"].map(
        lambda img: {
            "img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0
        },
        input_columns="image",
    )

    data = (
        train_partition["img"],
        train_partition["label"].astype(np.uint32),
        test_partition["img"],
        test_partition["label"].astype(np.uint32),
    )

    train_images, train_labels, test_images, test_labels = map(mx.array, data)


The Model
---------

We define the model as in the centralized MLX example, it's a simple MLP:

.. code-block:: python

    class MLP(nn.Module):
        """A simple MLP."""

        def __init__(
            self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
        ):
            super().__init__()
            layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
            self.layers = [
                nn.Linear(idim, odim)
                for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
            ]

        def __call__(self, x):
            for l in self.layers[:-1]:
                x = mx.maximum(l(x), 0.0)
            return self.layers[-1](x)

We also define some utility functions to test our model and to iterate over batches.

.. code-block:: python

    def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))


    def eval_fn(model, X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)


    def batch_iterate(batch_size, X, y):
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]


The ClientApp
-------------

The main changes we have to make to use `MLX` with `Flower` will be found in
the `get_params` and `set_params` functions. Indeed, MLX doesn't
provide an easy way to convert the model parameters into a list of `np.array`s
(the format we need for the serialization of the messages to work).

The way MLX stores its parameters is as follows:

.. code-block:: shell

    { 
    "layers": [
        {"weight": mlx.core.array, "bias": mlx.core.array},
        {"weight": mlx.core.array, "bias": mlx.core.array},
        ...,
        {"weight": mlx.core.array, "bias": mlx.core.array}
    ]
    }

Therefore, to get our list of `np.array`s, we need to extract each array and
convert them into a NumPy array:

.. code-block:: python

    def get_params(model):
        layers = model.parameters()["layers"]
        return [np.array(val) for layer in layers for _, val in layer.items()]


For the `set_params` function, we perform the reverse operation. We receive
a list of NumPy arrays and want to convert them into MLX parameters. Therefore, we
iterate through pairs of parameters and assign them to the `weight` and `bias`
keys of each layer dict:

.. code-block:: python

    def set_params(model, parameters):
    new_params = {}
    new_params["layers"] = [
        {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
        for i in range(0, len(parameters), 2)
    ]
    model.update(new_params)


The rest of the functionality is directly inspired by the centralized case. The `fit()`
method in the client trains the model using the local dataset:

.. code-block:: python

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                _, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return self.get_parameters(config={}), len(self.train_images), {}


Here, after updating the parameters, we perform the training as in the
centralized case, and return the new parameters.

And for the `evaluate` method of the client:

.. code-block:: python

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


We also begin by updating the parameters with the ones sent by the server, and
then we compute the loss and accuracy using the functions defined above. In the
constructor of the `FlowerClient` we instantiate the `MLP` model as well as other
components such as the optimizer.

Putting everything together we have:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(
            self,
            data,
            num_layers,
            hidden_dim,
            num_classes,
            batch_size,
            learning_rate,
            num_epochs,
        ):
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs

            self.train_images, self.train_labels, self.test_images, self.test_labels = data
            self.model = MLP(
                num_layers, self.train_images.shape[-1], hidden_dim, num_classes
            )
            self.optimizer = optim.SGD(learning_rate=learning_rate)
            self.loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            self.num_epochs = num_epochs
            self.batch_size = batch_size

        def get_parameters(self, config):
            return get_params(self.model)

        def set_parameters(self, parameters):
            set_params(self.model, parameters)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            for _ in range(self.num_epochs):
                for X, y in batch_iterate(
                    self.batch_size, self.train_images, self.train_labels
                ):
                    _, grads = self.loss_and_grad_fn(self.model, X, y)
                    self.optimizer.update(self.model, grads)
                    mx.eval(self.model.parameters(), self.optimizer.state)
            return self.get_parameters(config={}), len(self.train_images), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            accuracy = eval_fn(self.model, self.test_images, self.test_labels)
            loss = loss_fn(self.model, self.test_images, self.test_labels)
            return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}


Finally, we can construct a `ClientApp` using the `FlowerClient` defined above by means of a `client_fn` callback:

.. code-block:: python

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data = load_data(partition_id, num_partitions)

        num_layers = context.run_config["num-layers"]
        hidden_dim = context.run_config["hidden-dim"]
        num_classes = 10
        batch_size = context.run_config["batch-size"]
        learning_rate = context.run_config["lr"]
        num_epochs = context.run_config["local-epochs"]

        # Return Client instance
        return FlowerClient(
            data, num_layers, hidden_dim, num_classes, batch_size, learning_rate, num_epochs
        ).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn)

The ServerApp
-------------

To construct a `ServerApp` we define a `server_fn()` callback with an identical signature
to that of `client_fn()` but the return type is `ServerAppComponents <https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html#serverappcomponents>`_ as opposed to a `Client <https://flower.ai/docs/framework/ref-api/flwr.client.Client.html#client>`_. In this example we use the `FedAvg` strategy.

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Define strategy
        strategy = FedAvg()
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)


Congratulations!
You've successfully built and run your first federated learning system.
The `source code <https://github.com/adap/flower/blob/main/examples/quickstart-mlx/client.py>`_ of the extended version of this tutorial can be found in :code:`examples/quickstart-mlx`.
