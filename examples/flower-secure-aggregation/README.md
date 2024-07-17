---
title: Simple Secure Aggregation with Flower Example
tags: [basic, vision, fds]
dataset: []
framework: [numpy]
---

# Secure aggregation with Flower (the SecAgg+ protocol)

The following steps describe how to use Secure Aggregation in flower, with `ClientApp` using `secaggplus_mod` and `ServerApp` using `SecAggPlusWorkflow`.

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

Run:

```bash
flwr run
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config 'num_server_rounds=5'
```

### Alternative wasy of running the example

TODO: point to docs

## Amend the example for practical usage

For real-world applications, modify the `workflow` in `secaggexample/server_app.py` as follows:

```python
workflow = fl.server.workflow.DefaultWorkflow(
    fit_workflow=SecAggPlusWorkflow(
        num_shares=<number of shares>,
        reconstruction_threshold=<reconstruction threshold>,
    )
)
```
