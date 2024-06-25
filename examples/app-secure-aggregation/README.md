---
title: "Simple Flower Example using PyTorch"
url: https://pytorch.org/
labels: [basic, vision, fds]
dataset: [CIFAR-10]
---

# Secure aggregation with Flower (the SecAgg+ protocol) 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to use Secure Aggregation in flower, with `ClientApp` using `secaggplus_mod` and `ServerApp` using `SecAggPlusWorkflow`.

## Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
├── client.py               # Client application using `secaggplus_mod`
├── server.py               # Server application using `SecAggPlusWorkflow`
├── workflow_with_log.py    # Augmented `SecAggPlusWorkflow`
├── run.sh                  # Quick start script
├── pyproject.toml          # Project dependencies (poetry)
└── requirements.txt        # Project dependencies (pip)
```

## Installing dependencies

Project dependencies (such as and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

If you don't see any errors you're good to go!

## Run the example with one command (recommended)

```bash
./run.sh
```

## Run the example with the simulation engine

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 5
```

## Alternatively, run the example (in 7 terminal windows)

Start the Flower Superlink in one terminal window:

```bash
flower-superlink --insecure
```

Start 5 Flower `ClientApp` in 5 separate terminal windows:

```bash
flower-client-app client:app --insecure
```

Start the Flower `ServerApp`:

```bash
flower-server-app server:app --insecure --verbose
```

## Amend the example for practical usage

For real-world applications, modify the `workflow` in `server.py` as follows:

```python
workflow = fl.server.workflow.DefaultWorkflow(
    fit_workflow=SecAggPlusWorkflow(
        num_shares=<number of shares>,
        reconstruction_threshold=<reconstruction threshold>,
    )
)
```
