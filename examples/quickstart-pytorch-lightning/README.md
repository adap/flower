---
tags: [quickstart, vision, fds]
dataset: [MNIST]
framework: [lightning]
---

# Flower Example using PyTorch Lightning

This introductory example to Flower uses PyTorch Lightning, but deep knowledge of PyTorch Lightning is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

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
├── README.md
├── pyproject.toml
└── pytorchlightning_example
    ├── client_app.py
    └── server_app.py
```

### Installing Dependencies

Project dependencies are defined in `pyproject.toml`.
You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
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
flwr run --run-config 'num_server_rounds=5'
```

### Alternative way of running the example
