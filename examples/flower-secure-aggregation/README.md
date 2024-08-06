---
title: Simple Secure Aggregation with Flower Example
tags: [advanced, secure_aggregation, privacy]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Secure aggregation with Flower (the SecAgg+ protocol)

The following steps describe how to use Secure Aggregation in flower, with `ClientApp` using `secaggplus_mod` and `ServerApp` using `SecAggPlusWorkflowWithLogs`, which is a subclass of `SecAggPlusWorkflow` that includes more detailed logging specifically designed for this example.

## Project Setup

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
      |        ├── __init__.py
      |        ├── client_app.py    # defines your ClientApp
      |        ├── server_app.py    # defines your ServerApp
      |        ├── task.py
      |        └── workflow_with_log.py
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

Run:

```bash
flwr run
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config 'num_server_rounds=5'
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.

## Amend the example for practical usage

To adapt the example for real-world applications, follow these steps:

1. Set `IS_DEMO` to `False` in `./secaggexample/task.py`.
2. Adjust the `num-shares` and `reconstruction-threshold` settings in `./pyproject.toml` to suit your requirements.
