---
title: Federated Learning with MLX and Flower (Quickstart Example)
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Federated Learning with MLX and Flower (Quickstart Example)

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but you don't need deep knowledge of MLX to run it. The example will help you understand how to adapt Flower to your specific use case, and running it is quite straightforward.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple Silicon. In this example, we will train a simple 2-layer MLP on the [MNIST](https://huggingface.co/datasets/ylecun/mnist) dataset (handwritten digits recognition). The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/).

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

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart MLX tutorial](https://flower.ai/docs/framework/tutorial-quickstart-mlx.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates.
