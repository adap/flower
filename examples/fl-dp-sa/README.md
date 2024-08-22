---
tags: [basic, vision, fds]
dataset: [MNIST]
framework: [torch, torchvision]
---

# Example of Flower App with DP and SA

This is a simple example that utilizes central differential privacy with client-side fixed clipping and secure aggregation.
Note: This example is designed for a small number of rounds and is intended for demonstration purposes.

## Project Setup

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-dp-sa . && rm -rf flower && cd fl-dp-sa
```

This will create a new directory called `fl-dp-sa` containing the following files:

```shell
fl-dp-sa
|
├── fl-dp-sa
|   ├── __init__.py
|   ├── client_app.py    # Defines your ClientApp
|   ├── server_app.py    # Defines your ServerApp
|   ├── task.py          # Defines your model, training and data loading
├── pyproject.toml       # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fl-dp-sa` package.

```bash
pip install -e .
```
## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

The example uses the MNIST dataset with a total of 100 clients, with 20 clients sampled in each round. The hyperparameters for DP and SecAgg are specified in `server.py`.

```shell
flower-simulation --server-app fl_dp_sa.server:app --client-app fl_dp_sa.client:app --num-supernodes 100
```
