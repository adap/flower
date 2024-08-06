---
tags: [quickstart, vision, fds]
dataset: [MNIST]
framework: [lightning]
---

# Federated Learning with PyTorch Lightning and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch Lightning, but deep knowledge of PyTorch Lightning is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset. The model being federated is a lightweight AutoEncoder as presented in [Lightning in 15 minutes](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) tutorial.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pytorch-lightning . \
        && rm -rf _tmp && cd quickstart-pytorch-lightning
```

This will create a new directory called `quickstart-pytorch-lightning` containing the following files:

```shell
quickstart-pytorch-lightning
├── pytorchlightning_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

# Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchlightning_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5,max-epochs=2
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
