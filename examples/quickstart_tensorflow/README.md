# Flower Example using Keras

Flower is easy to use by only starting the Flower server and the Flower clients.

It is recommended to use [pyenv](https://github.com/pyenv/pyenv)/[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) and [poetry](https://python-poetry.org/docs/) to ensure the right version of libraries.

After installing both tools you can start to set up everything.  

## Setup your virtual environment

First, you should setup your virtualenv and  install [Python Version 3.6](https://docs.python.org/3.6/) or above:

```shell
pyenv install 3.7.9
```

Create a virtualenv with:

```shell
pyenv virtualenv 3.7.9 keras-federated-3.7.9
```

Activate the virtualenv by creating a `.python-version` with the content:

```shell
keras-federated-3.7.9
```

## Run Keras Federated

Clone the flower examples to your virtualenv:

```shell
git clone https://github.com/adap/.....
```

You have different files available:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- run-clients.sh
-- run-server.sh
```

The `pyproject.toml` and `poetry.lock` orchestrate the project dependencies. Simply run poetry to install all required dependencies with:

```shell
poetry install
```

After installing all required libaries you can start the Flower server

```shell
./run-server.sh
```

Open a new terminal and start 2 Flower clients with

```shell
./run-client.sh
```

You will see that Keras is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart_tensorflow.html) for a detailed explanation of the code.
