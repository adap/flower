---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST | https://huggingface.co/datasets/ylecun/mnist]
framework: [MLX | https://ml-explore.github.io/mlx/build/html/index.html]
---

# Flower Example using MLX

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but deep knowledge of MLX is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon.

In this example, we will train a simple 2 layers MLP on MNIST data (handwritten digits recognition).

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart-mlx . && rm -rf _tmp && cd quickstart-mlx
```

This will create a new directory called `quickstart-mlx` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- run.sh
-- README.md
```

### Installing Dependencies

Project dependencies (such as `mlx` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with MLX and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the
following commands.

Start a first client in the first terminal:

```shell
python3 client.py --partition-id 0
```

And another one in the second terminal:

```shell
python3 client.py --partition-id 1
```

If you want to utilize your GPU, you can use the `--gpu` argument:

```shell
python3 client.py --gpu --partition-id 2
```

Note that you can start many more clients if you want, but each will have to be in its own terminal.

You will see that MLX is starting a federated training. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-mlx) for a detailed explanation.

## Explanations

This example is a federated version of the centralized case that can be found
[here](https://github.com/ml-explore/mlx-examples/tree/main/mnist).

### The data

We will use `flwr_datasets` to easily download and partition the `MNIST` dataset:

```python
fds = FederatedDataset(dataset="mnist", partitioners={"train": 3})
partition = fds.load_partition(partition_id = args.partition_id)
partition_splits = partition.train_test_split(test_size=0.2)

partition_splits['train'].set_format("numpy")
partition_splits['test'].set_format("numpy")

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

train_images, train_labels, test_images, test_labels = map(mlx.core.array, data)
```

### The model

We define the model as in the centralized mlx example, it's a simple MLP:

```python
class MLP(mlx.nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            mlx.nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mlx.core.maximum(l(x), 0.0)
        return self.layers[-1](x)

```

We also define some utility functions to test our model and to iterate over batches.

```python
def loss_fn(model, X, y):
    return mlx.core.mean(mlx.nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mlx.core.mean(mlx.core.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mlx.core.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

```

### The client

The main changes we have to make to use `MLX` with `Flower` will be found in
the `get_parameters` and `set_parameters` functions. Indeed, MLX doesn't
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
convert them into a numpy array:

```python
def get_parameters(self, config):
    layers = self.model.parameters()["layers"]
    return [np.array(val) for layer in layers for _, val in layer.items()]
```

For the `set_parameters` function, we perform the reverse operation. We receive
a list of arrays and want to convert them into MLX parameters. Therefore, we
iterate through pairs of parameters and assign them to the `weight` and `bias`
keys of each layer dict:

```python
def set_parameters(self, parameters):
    new_params = {}
    new_params["layers"] = [
        {"weight": mlx.core.array(parameters[i]), "bias": mlx.core.array(parameters[i + 1])}
        for i in range(0, len(parameters), 2)
    ]
    self.model.update(new_params)
```

The rest of the functions are directly inspired by the centralized case:

```python
def fit(self, parameters, config):
    self.set_parameters(parameters)
    for _ in range(self.num_epochs):
        for X, y in batch_iterate(
            self.batch_size, self.train_images, self.train_labels
        ):
            loss, grads = self.loss_and_grad_fn(self.model, X, y)
            self.optimizer.update(self.model, grads)
            mlx.core.eval(self.model.parameters(), self.optimizer.state)
    return self.get_parameters(config={}), len(self.train_images), {}
```

Here, after updating the parameters, we perform the training as in the
centralized case, and return the new parameters.

And for the `evaluate` function:

```python
def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    accuracy = eval_fn(self.model, self.test_images, self.test_labels)
    loss = loss_fn(self.model, self.test_images, self.test_labels)
    return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}
```

We also begin by updating the parameters with the ones sent by the server, and
then we compute the loss and accuracy using the functions defined above.

Putting everything together we have:

```python
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, model, optim, loss_and_grad_fn, data, num_epochs, batch_size
    ) -> None:
        self.model = model
        self.optimizer = optim
        self.loss_and_grad_fn = loss_and_grad_fn
        self.train_images, self.train_labels, self.test_images, self.test_labels = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def get_parameters(self, config):
        layers = self.model.parameters()["layers"]
        return [np.array(val) for layer in layers for _, val in layer.items()]

    def set_parameters(self, parameters):
        new_params = {}
        new_params["layers"] = [
            {"weight": mlx.core.array(parameters[i]), "bias": mlx.core.array(parameters[i + 1])}
            for i in range(0, len(parameters), 2)
        ]
        self.model.update(new_params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(self.num_epochs):
            for X, y in batch_iterate(
                self.batch_size, self.train_images, self.train_labels
            ):
                loss, grads = self.loss_and_grad_fn(self.model, X, y)
                self.optimizer.update(self.model, grads)
                mlx.core.eval(self.model.parameters(), self.optimizer.state)
        return self.get_parameters(config={}), len(self.train_images), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = eval_fn(self.model, self.test_images, self.test_labels)
        loss = loss_fn(self.model, self.test_images, self.test_labels)
        return loss.item(), len(self.test_images), {"accuracy": accuracy.item()}
```

And as you can see, with only a few lines of code, our client is ready! Before
we can instantiate it, we need to define a few variables:

```python
num_layers = 2
hidden_dim = 32
num_classes = 10
batch_size = 256
num_epochs = 1
learning_rate = 1e-1

model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)

loss_and_grad_fn = mlx.nn.value_and_grad(model, loss_fn)
optimizer = mlx.optimizers.SGD(learning_rate=learning_rate)
```

Finally, we can instantiate it by using the `start_client` function:

```python
# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(
        model,
        optimizer,
        loss_and_grad_fn,
        (train_images, train_labels, test_images, test_labels),
        num_epochs,
        batch_size,
    ).to_client(),
)
```

### The server

On the server side, we don't need to add anything in particular. The
`weighted_average` function is just there to be able to aggregate the results
and have an accuracy at the end.

```python
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```
