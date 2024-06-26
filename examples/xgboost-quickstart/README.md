---
title: Flower Example using PyTorch
labels: [comprehensive, classification, tabular]
dataset: [HIGGS]
framework: [xgboost]
---

# Flower Example using XGBoost

This example demonstrates how to perform EXtreme Gradient Boosting (XGBoost) within Flower using `xgboost` package.
We use [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs) dataset for this example to perform a binary classification task.
Tree-based with bagging method is used for aggregation on the server.

This project provides a minimal code example to enable you to get started quickly. For a more comprehensive code example, take a look at [xgboost-comprehensive](https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive).

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/xgboost-quickstart . && rm -rf flower && cd xgboost-quickstart
```

This will create a new directory called `xgboost-quickstart` containing the following files:

```
-- README.md         <- Your're reading this right now
-- server.py         <- Defines the server-side logic
-- client.py         <- Defines the client-side logic
-- run.sh            <- Commands to run experiments
-- pyproject.toml    <- Example dependencies
```

### Installing Dependencies

Project dependencies (such as `xgboost` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Run Federated Learning with XGBoost and Flower

Afterwards you are ready to start the Flower server as well as the clients.
You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning.
To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id=0
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id=1
```

You will see that XGBoost is starting a federated training.

Alternatively, you can use `run.sh` to run the same experiment in a single terminal as follows:

```shell
poetry run ./run.sh
```

Look at the [code](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart)
and [tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html) for a detailed explanation.
