# Flower Example using XGBoost

This example demonstrates how to perform EXtreme Gradient Boosting (XGBoost) within Flower using `xgboost` package.
Tree-based with bagging method is used for aggregation on the server.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-xgboost . && rm -rf flower && cd quickstart-xgboost
```

This will create a new directory called `quickstart-xgboost` containing the following files:

```
-- README.md         <- Your're reading this right now
-- server.py         <- Defines the server-side logic
-- strategy.py       <- Defines the tree-based bagging aggregation
-- client.py         <- Defines the client-side logic
-- dataset.py        <- Defines the functions of data loading and partitioning
-- pyproject.toml    <- Example dependencies (if you use Poetry)
-- requirements.txt  <- Example dependencies
```

### Installing Dependencies

Project dependencies (such as `xgboost` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

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
python3 client.py node-id=0
```

Start client 2 in the second terminal:

```shell
python3 client.py node-id=1
```

You will see that XGBoost is starting a federated training.

Alternatively, you can use `run.sh` to run the same experiment in a single terminal as follows:

```shell
bash run.sh
```

Besides, we provide options to customise the experimental settings, including data partitioning and centralised/distributed evaluation (see `utils.py`).
Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-xgboost)
and [tutorial](https://flower.dev/docs/framework/tutorial-quickstart-xgboost.html) for a detailed explanation.
