---
tags: [DP, SecAgg, vision, fds]
dataset: [MNIST]
framework: [torch, torchvision]
---

# Flower Example on MNIST with Differential Privacy and Secure Aggregation

This example demonstrates a federated learning setup using the Flower, incorporating central differential privacy (DP) with client-side fixed clipping and secure aggregation (SA). It is intended for a small number of rounds for demonstration purposes.

This example is similar to the [quickstart-pytorch example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) and extends it by integrating central differential privacy and secure aggregation. For more details on differential privacy and secure aggregation in Flower, please refer to the documentation [here](https://flower.ai/docs/framework/how-to-use-differential-privacy.html) and [here](https://flower.ai/docs/framework/contributor-ref-secure-aggregation-protocols.html).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-dp-sa . && rm -rf flower && cd fl-dp-sa
```

This will create a new directory called `fl-dp-sa` containing the following files:

```shell
fl-dp-sa
├── fl_dp_sa
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fl_dp_sa` package.

```shell
# From a new python environment, run:
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
flwr run . --run-config "noise-multiplier=0.1 clipping-norm=5"
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.
