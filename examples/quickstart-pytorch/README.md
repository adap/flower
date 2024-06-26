---
title: Simple Flower Example using PyTorch
url: https://pytorch.org/
labels: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Example using PyTorch

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-pytorch . && rm -rf flower && cd quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

______________________________________________________________________

## Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. We need to specify the partition id to
use different partitions of the data on different nodes.  To do so simply open two more terminal windows and run the
following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1
```

You will see that PyTorch is starting a federated training. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) for a detailed explanation.

______________________________________________________________________

## Run Federated Learning with PyTorch and `Flower Next`

### 1. Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure
```

### 2. Start the long-running Flower clients (SuperNodes)

Start 2 Flower `SuperNodes` in 2 separate terminal windows, using:

```bash
flower-client-app client:app --insecure
```

### 3. Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```
