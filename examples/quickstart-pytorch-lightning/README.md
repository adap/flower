---
title: Simple Flower Example using PyTorch-Lightning
tags: [quickstart, vision, fds]
dataset: [MNIST | https://huggingface.co/datasets/ylecun/mnist]
framework: [lightning | https://lightning.ai/docs/pytorch/stable/]
---

# Flower Example using PyTorch Lightning

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch Lightning is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-pytorch-lightning . && rm -rf flower && cd quickstart-pytorch-lightning
```

This will create a new directory called `quickstart-pytorch-lightning` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py  # client-side code
-- server.py # server-side code (including the strategy)
-- README.md
-- run.sh # runs server, then two clients
-- mnist.py # run a centralised version of this example
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. We need to specify the partition id to
use different partitions of the data on different nodes.  To do so simply open two more terminal windows and run the
following commands.

Start client 1 in the first terminal:

```shell
python client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
python client.py --partition-id 1
```

You will see that PyTorch is starting a federated training. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) for a detailed explanation.
