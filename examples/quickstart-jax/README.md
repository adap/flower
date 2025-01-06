---
tags: [quickstart, linear regression]
dataset: [Synthetic]
framework: [JAX, FLAX]
---

# Federated Learning with JAX and Flower (Quickstart Example)

This introductory example to Flower uses JAX, but deep knowledge of JAX is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [FLAX](https://flax.readthedocs.io/en/latest/index.html) to define and train a small CNN model. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MINST dataset.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-jax . \
        && rm -rf _tmp \
        && cd quickstart-jax
```

This will create a new directory called `quickstart-jax` with the following structure:

```shell
quickstart-jax
├── jaxexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `jaxexample` package.

```bash
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
flwr run . --run-config "num-server-rounds=5 batch-size=32"
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart JAX tutorial](https://flower.ai/docs/framework/tutorial-quickstart-jax.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
