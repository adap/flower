---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Flower Example using MLX

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but deep knowledge of MLX is not necessary to run the example. This example will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple Silicon. In this example, we will train a simple 2 layers MLP on [MNIST](https://huggingface.co/datasets/ylecun/mnist) data (handwritten digits recognition) that's downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-mlx . \
        && rm -rf _tmp \
        && cd quickstart-mlx
```

This will create a new directory called `quickstart-mlx` with the following structure:

```shell
quickstart-mlx
├── mlxexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `mlxexample` package.

```bash
pip install -e .
```

## Run the project

You can run your `ClientApp` and `ServerApp` in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ model as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5,learning-rate=0.05
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Explanation

This example is a federated version of the centralized case that can be found
[here](https://github.com/ml-explore/mlx-examples/tree/main/mnist).

### The data

We will use `flwr_datasets` to easily download and partition the `MNIST` dataset. In this example you'll make use of the [IidPartitioner](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner) to generate `num_partitions` partitions. You can choose [other partitioners](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html) available in Flower Datasets:

```python
partitioner = IidPartitioner(num_partitions=num_partitions)
fds = FederatedDataset(
    dataset="ylecun/mnist",
    partitioners={"train": partitioner},
    trust_remote_code=True,
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
```

### The model

We define the model as in the centralized MLX example, it's a simple MLP:

```python
class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int = 10
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
```

We also define some utility functions to test our model and to iterate over batches.

```python
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]
```

### The ClientApp

The main changes we have to make to use `MLX` with `Flower` will be found in
the `get_params` and `set_params` functions. Indeed, MLX doesn't
provide an easy way to convert the model parameters into a list of `np.array`s
(the format we need for the serialization of the messages to work).

The way MLX stores its parameters is as follows:

```
{ 
  "layers": [
    {"weight": mlx.core.array, "bias": mlx.core.array},
    {"weight": mlx.core.array, "bias": mlx.core.array},
    ...,
    {"weight": mlx.core.array, "bias": mlx.core.array}
  ]
}
```

Therefore, to get our list of `np.array`s, we need to extract each array and
convert them into a NumPy array:

```python
def get_params(model):
    layers = model.parameters()["layers"]
    return [np.array(val) for layer in layers for _, val in layer.items()]
```

For the `set_params` function, we perform the reverse operation. We receive
a list of NumPy arrays and want to convert them into MLX parameters. Therefore, we
iterate through pairs of parameters and assign them to the `weight` and `bias`
keys of each layer dict:

```python
def set_params(model, parameters):
    new_params = {}
    new_params["layers"] = [
        {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
        for i in range(0, len(parameters), 2)
    ]
    model.update(new_params)
```

The rest of the functionality is directly inspired by the centralized case. The `fit()`
method in the client trains the model using the local dataset:

```python
def fit(self, parameters, config):
    set_params(self.model, parameters)
    for _ in range(self.num_epochs):
        for X, y in batch_iterate(
            self.batch_size, self.train_images, self.train_labels
        ):
            _, grads = self.loss_and_grad_fn(self.model, X, y)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
    return self.get_parameters(config={}), len(self.train_images), {}
```

Here, after updating the parameters, we perform the training as in the
centralized case, and return the new parameters.

And for the `evaluate` method of the client:

```python
def evaluate(self, parameters, config):
    set_params(self.model, parameters)
    accuracy = eval_fn(self.model, self.test_images, self.test_labels)
    loss = loss_fn(self.model, self.test_images, self.test_labels)
    return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}
```

We also begin by updating the parameters with the ones sent by the server, and
then we compute the loss and accuracy using the functions defined above. In the
constructor of the `FlowerClient` we instantiate the `MLP` model as well as other
components such as the optimizer.

Putting everything together we have:

```python
class FlowerClient(NumPyClient):
    def __init__(self, num_layers, hidden_dim, batch_size, learning_rate, data):
        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.model = MLP(
            num_layers,
            self.train_images.shape[-1],
            hidden_dim,
        )
        self.optimizer = optim.SGD(learning_rate=learning_rate)
        self.loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
        self.num_epochs = 1
        self.batch_size = batch_size

    def get_parameters(self, config):
        """Return the parameters of the model of this client."""
        return get_params(self.model)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_params(self.model, parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                _, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
        return self.get_parameters(config={}), len(self.train_images), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_params(self.model, parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}
```

Finally, we can construct a `ClientApp` using the `FlowerClient` defined above by means of a `client_fn` callback:

```python
# Start Flower client
def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    num_layers = context.run_config["num-layers"]
    hidden_dim = context.run_config["hidden-dim"]
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(num_layers, hidden_dim, batch_size, lr, data).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)
```

### The ServerApp

To construct a `ServerApp` we define a `server_fn()` callback with an identical signature
to that of `client_fn()` but the return type is [ServerAppComponents](https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html#serverappcomponents) as opposed to a [`Client`](https://flower.ai/docs/framework/ref-api/flwr.client.Client.html#client). In this example we use the `FedAvg` strategy and pass a callback to aggregate the validation accuracies
returned by clients sampled in an _evaluate_ round.

```python
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Define the strategy
    fraction_eval = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_evaluate=fraction_eval,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
```
