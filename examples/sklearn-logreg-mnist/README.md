---
tags: [basic, vision, logistic regression, fds]
dataset: [MNIST]
framework: [scikit-learn]
---

# Flower Logistic Regression Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/sklearn-logreg-mnist . \
		&& rm -rf _tmp && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` with the following structure:

```shell
sklearn-logreg-mnist
├── README.md
├── pyproject.toml      # builds your project, includes dependencies and configs
└── sklearn_example
    ├── client_app.py   # defines your ClientApp
    └── server_app.py   # defines your ServerApp
```

## Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `sklearn_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make
use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
