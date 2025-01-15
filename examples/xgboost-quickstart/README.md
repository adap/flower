---
tags: [quickstart, classification, tabular]
dataset: [HIGGS]
framework: [xgboost]
---

# Federated Learning with XGBoost and Flower (Quickstart Example)

This example demonstrates how to perform EXtreme Gradient Boosting (XGBoost) within Flower using `xgboost` package.
We use [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) dataset for this example to perform a binary classification task.
Tree-based with bagging method is used for aggregation on the server.

This project provides a minimal code example to enable you to get started quickly. For a more comprehensive code example, take a look at [xgboost-comprehensive](https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/xgboost-quickstart . \
        && rm -rf _tmp \
        && cd xgboost-quickstart
```

This will create a new directory called `xgboost-quickstart` with the following structure:

```shell
xgboost-quickstart
├── xgboost_quickstart
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your utilities and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `xgboost_quickstart` package.

```bash
pip install -e .
```

> \[!NOTE\]
> For MacOSX users, you may need to additionally run `brew install libomp` to install OpenMP runtime.

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!NOTE\]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 params.eta=0.05"
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart XGBoost tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
