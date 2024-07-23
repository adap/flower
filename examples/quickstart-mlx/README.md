---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Flower Example using MLX

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but deep knowledge of MLX is not necessary to run the example. This example will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon.

In this example, we will train a simple 2 layers MLP on MNIST data (handwritten digits recognition) that's downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/)

## Project Setup

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
              && mv _tmp/examples/quickstart-mlx . \
              && rm -rf _tmp && cd quickstart-mlx
```

This will create a new directory called `quickstart-mlx` containing the
following files:

```shell
quickstart-mlx
      |
      ├── mlxexample
      |        ├── __init__.py
      |        ├── client_app.py    # defines your ClientApp
      |        ├── server_app.py    # defines your ServerApp
      |        └── task.py
      ├── pyproject.toml            # builds your FAB, includes dependencies and configs
      └── README.md
```

## Install dependencies

```bash
pip install .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make
use of the Simluation Engine. Refer to alternative ways of running your
Flower application including Deployment, with TLS certificates, or with
Docker later in this readme.

### Run with the Simulation Engine

Run:

```bash
flwr run
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config hidden_dim=64,num_server_rounds=5
```

### Alternative wasy of running the example

TODO: point to docs
