---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Flower Example using MLX

> \[!TIP\]
> An example created from `flwr new`'s `MLX` template with updated `client_fn` signature.


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
       |        ├── client_app.py
       |        ├── server_app.py
       |        └── task.py
       ├── pyproject.toml            # contains user config
       |                             # uses flwr-nightly[simulation]
       └── README.md
```

## Install dependencies

```bash
pip install .
```

## Run (Simulation Engine)

```bash
flwr run
```

## Run (Deployment Engine)

### Start the SuperExec

```bash
flower-superexec flwr.superexec.deployment:executor --insecure
```

### Start the SuperLink

```bash
flower-superlink --insecure
```

### Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
flower-supernode mlxexample.client:app --insecure --partition-id=0
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode mlxexample.client:app --insecure --partition-id=1
```

### Start the Run

```bash
flwr run --use-superexec
```
