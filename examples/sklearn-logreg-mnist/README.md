---
tags: [basic, vision, logistic regression, fds]
dataset: [MNIST]
framework: [scikit-learn]
---

# Flower Logistic Regression Example using scikit-learn and Flower (Quickstart Example)

This example of Flower uses `scikit-learn`'s [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/sklearn-logreg-mnist . \
		&& rm -rf _tmp && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` with the following structure:

```shell
sklearn-logreg-mnist
├── README.md
├── pyproject.toml      # Project metadata like dependencies and configs
└── sklearn_example
    ├── __init__.py
    ├── client_app.py   # Defines your ClientApp
    ├── server_app.py   # Defines your ServerApp
    └── task.py         # Defines your model, training and data loading
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `sklearn_example` package.

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
flwr run . --run-config "num-server-rounds=5 fraction-fit=0.25"
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-scikitlearn.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
