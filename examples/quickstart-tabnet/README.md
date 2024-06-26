---
title: Simple Flower Example using Tabnet
labels: [quickstart, tabular]
dataset: [iris]
framework: [tabnet]
---

# Flower TabNet Example using TensorFlow

This introductory example to Flower uses Keras but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understanding how to adapt Flower to your use-cases. You can learn more about TabNet from [paper](https://arxiv.org/abs/1908.07442) and its implementation using TensorFlow at [this repository](https://github.com/titu1994/tf-TabNet). Note also that the basis of this example using federated learning is the example from the repository above.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-tabnet . && rm -rf flower && cd quickstart-tabnet
```

This will create a new directory called `quickstart-tabnet` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Learning with TensorFlow/Keras and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
poetry run python client.py
```

Alternatively you can run all of it in one shell as follows:

```shell
poetry run python server.py &
poetry run python client.py &
poetry run python client.py
```
