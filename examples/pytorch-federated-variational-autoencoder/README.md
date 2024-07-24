---
tags: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Example for Federated Variational Autoencoder using Pytorch

This example demonstrates how a variational autoencoder (VAE) can be trained in a federated way using the Flower framework.

## Project Setup

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
              && mv _tmp/examples/pytorch-federated-variational-autoencoder . \
              && rm -rf _tmp && cd pytorch-federated-variational-autoencoder
```

This will create a new directory called `pytorch-federated-variational-autoencoder`
following files:

```shell
pytorch-federated-variational-autoencoder
├── README.md
├── fedvaeexample
│   ├── __init__.py
│   ├── client_app.py   # defines your ClientApp
│   ├── models.py       # defines your model
│   └── server_app.py   # defines your ServerApp
└── pyproject.toml      # builds your project, includes dependencies and configs
```

## Install dependencies

Install the dependencies defined in `pyproject.toml` as well as the `custom_metrics_example` package.

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