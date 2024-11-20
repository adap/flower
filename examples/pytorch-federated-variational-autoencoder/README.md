---
tags: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Federated Variational Autoencoder with PyTorch and Flower

This example demonstrates how a variational autoencoder (VAE) can be trained in a federated way using the Flower framework. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
              && mv _tmp/examples/pytorch-federated-variational-autoencoder . \
              && rm -rf _tmp && cd pytorch-federated-variational-autoencoder
```

This will create a new directory called `pytorch-federated-variational-autoencoder` with the following structure:

```shell
pytorch-federated-variational-autoencoder
├── README.md
├── fedvaeexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
└── pyproject.toml      # Project metadata like dependencies and configs
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fedvaeexample` package.

```bash
pip install -e .
```

## Run the Project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

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
