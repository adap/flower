---
tags: [advanced, secure_aggregation, privacy]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Secure aggregation with Flower (the SecAgg+ protocol)

The following steps describe how to use Flower's built-in Secure Aggregation components. This example demonstrates how to apply `SecAgg+` to the same federated learning workload as in the [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example. The `ServerApp` uses the [`SecAggPlusWorkflow`](https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html#secaggplusworkflow) while `ClientApp` uses the [`secaggplus_mod`](https://flower.ai/docs/framework/ref-api/flwr.client.mod.secaggplus_mod.html#flwr.client.mod.secaggplus_mod). To introduce the various steps involved in `SecAgg+`, this example introduces as a sub-class of `SecAggPlusWorkflow` the `SecAggPlusWorkflowWithLogs`. It is enabled by default, but you can disable (see later in this readme).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
              && mv _tmp/examples/flower-secure-aggregation . \
              && rm -rf _tmp && cd flower-secure-aggregation
```

This will create a new directory called `flower-secure-aggregation` containing the
following files:

```shell
flower-secure-aggregation
|
├── secaggexample
|   ├── __init__.py
|   ├── client_app.py    # Defines your ClientApp
|   ├── server_app.py    # Defines your ServerApp
|   ├── task.py          # Defines your model, training and data loading
|   └── workflow_with_log.py # Defines a workflow used when `is-demo=true`
├── pyproject.toml       # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `secaggexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . --run-config num-server-rounds=5,learning-rate=0.25
```

To adapt the example for a practial usage, set `is-demo=false` like shown below. You might want to adjust the `num-shares` and `reconstruction-threshold` settings to suit your requirements. You can override those via `--run-config` as well.

```bash
flwr run . --run-config is-demo=false
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.
