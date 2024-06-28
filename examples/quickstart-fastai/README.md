---
title: Simple Flower Example using fastai
labels: [quickstart, vision]
dataset: [`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_]
framework: [`fastai <https://fast.ai>`_]
---

# Flower Example using fastai

This introductory example to Flower uses [fastai](https://www.fast.ai/), but deep knowledge of fastai is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart-fastai . && rm -rf _tmp && cd quickstart-fastai
```

This will create a new directory called `quickstart-fastai` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- run.sh
-- README.md
```

### Installing Dependencies

Project dependencies (such as `fastai` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Learning with fastai and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```

You will see that fastai is starting a federated training. For a more in-depth look, be sure to check out the code on our [repo](https://github.com/adap/flower/tree/main/examples/quickstart-fastai).
