---
tags: [basic, vision, logistic regression, fds]
dataset: [MNIST]
framework: [scikit-learn]
---

# Flower Logistic Regression Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/sklearn-logreg-mnist . \
		&& rm -rf _tmp && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` containing the following files:

```shell
├── sklearn_example
|   ├── __init__.py
|   ├── client_app.py
|   └── server_app.py
├── pyproject.toml
└── README.md 
```

### Installing Dependencies

Project dependencies are defined in `pyproject.toml`.
You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually.

### Run with the Simulation Engine

First, launch the [SuperExec](link-to-docs).

```bash
flower-superexec flwr.superexec.simulation:executor --insecure
```

In a new terminal, and after activating your Python environment:

```bash
flwr run --use-superexec
```

### Run with the Deployment Engine

Launch the the infrastructure needed for a `Run` to run. This means:
the [`SuperLink`](https://flower.ai/docs/framework/ref-api-cli.html#flower-superlink) and at least two [`SuperNode`](docs) instances.
You will need a few terminal windows (consider using [tmux](https://github.com/tmux/tmux/wiki)), remember
to activate your environment in each of them.

1. On a new terminal, launch the `SuperLink`:
   ```bash
   flower-superlink --insecure
   ```
1. On a new terminal, launch a `SuperNode`:
   ```bash
   flower-supernode --insecure --partition-id=0
   ```
1. On a new terminal, launch another `SuperNode`:
   ```bash
   flower-supernode --insecure --partition-id=1
   ```

With everything ready and idling, launch the [SuperExec](link-to-docs).

```bash
flower-superexec flwr.superexec.deployment:executor --insecure
```

Then,

```bash
flwr run --user-superexec
```

You will see that Flower is starting a federated training.
