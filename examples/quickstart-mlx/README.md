---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Flower Example using MLX

## Project Setup

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/<example-name> . \
		&& rm -rf _tmp && cd <example-name>
```

This will create a new directory called `quickstart-mlx` containing the
following files:

```shell
<quickstart-mlx>
       |
       ├── <mlxexample>
       |        ├── __init__.py
       |        ├── client_app.py    # defines your ClientApp
       |        ├── server_app.py    # defines your ServerApp
       |        └── task.py
       ├── pyproject.toml            # builds your FAB, includes dependencies and configs
       └── README.md
```

## Install dependencies

```bash
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

```bash
flwr run
```

### Alternativ
