---
title: Simple Flower Example using Jax
labels: [quickstart, linear regression]
dataset: [synthetic]
framework: [JAX]
---

# JAX: From Centralized To Federated

This example demonstrates how an already existing centralized JAX-based machine learning project can be federated with Flower.

This introductory example for Flower uses JAX, but you're not required to be a JAX expert to run the example. The example will help you to understand how Flower can be used to build federated learning use cases based on an existing JAX project.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-jax . && rm -rf flower && cd quickstart-jax
```

This will create a new directory called `quickstart-jax`, containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- jax_training.py
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `jax` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run JAX Federated

This JAX example is based on the [Linear Regression with JAX](https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html) tutorial and uses a sklearn dataset (generating a random dataset for a regression problem). Feel free to consult the tutorial if you want to get a better understanding of JAX. If you play around with the dataset, please keep in mind that the data samples are generated randomly depending on the settings being done while calling the dataset function. Please checkout out the [scikit-learn tutorial for further information](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html). The file `jax_training.py` contains all the steps that are described in the tutorial. It loads the train and test dataset and a linear regression model, trains the model with the training set, and evaluates the trained model on the test set.

The only things we need are a simple Flower server (in `server.py`) and a Flower client (in `client.py`). The Flower client basically takes model and training code tells Flower how to call it.

Start the server in a terminal as follows:

```shell
python3 server.py
```

Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```

You are now training a JAX-based linear regression model, federated across two clients. The setup is of course simplified since both clients hold a similar dataset, but you can now continue with your own explorations. How about changing from a linear regression to a more sophisticated model? How about adding more clients?
