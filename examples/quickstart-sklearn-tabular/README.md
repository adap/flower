---
tags: [quickstart, tabular, fds]
dataset: [Iris]
framework: [scikit-learn]
---

# Flower Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system on
"iris" (tabular) dataset.
It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to
download, partition and preprocess the dataset.

## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-sklearn-tabular . && rm -rf flower && cd quickstart-sklearn-tabular
```

This will create a new directory called `quickstart-sklearn-tabular` with the following structure:

```shell
quickstart-sklearn-tabular
├── example
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
flwr run . --run-config penalty="l1"
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Explanation

This example is a federated version of the centralized case that can be found
[here](https://github.com/ml-explore/mlx-examples/tree/main/mnist).
