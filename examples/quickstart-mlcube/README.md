---
title: Flower Example using TensorFlow/Keras + MLCube 
labels: [quickstart, vision, deployment]
dataset: [MNIST | https://huggingface.co/datasets/ylecun/mnist]
framework: [tensorflow | https://www.tensorflow.org/, Keras]
---

# Flower Example using TensorFlow/Keras + MLCube

This introductory example to Flower uses MLCube together with Keras, but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use-cases with MLCube. Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell, which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart-mlcube . && rm -rf flower && cd quickstart-mlcube
```

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly, you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors, you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

To verify that everything works correctly, you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors, you're good to go!

#### Docker

For the MLCube setup you will need to install Docker on your system. Please refer to the [Docker install guide](https://docs.docker.com/get-docker/) on how to do that.

#### MLCube

For the MLCube setup, we have prepared a script that you can execute in your shell using:

```shell
./dev/setup.sh
```

## Run Federated Learning with TensorFlow/Keras in MLCube with Flower

Afterward, you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
./dev/server.sh
```

Now you are ready to start the clients. We have prepared a simple script called `client.sh`, which accepts a CLIENT_ID and can be executed as in:

```shell
# Shell 1
./dev/client.sh 1
```

```shell
# Shell 2
./dev/client.sh 2
```

Congrats! You have just run a Federated Learning experiment using TensorFlow/Keras in MLCube using Flower for federation.

## Background

Wondering how this works? Most of the interaction with MLCube happens in `mlcube_utils.py`, which reads and writes to the file system. It also provides a function called `run_task`, which invokes `mlcube_docker run ...` to execute the appropriate task. The custom client we have implemented in `client.py` will run the MLCube 'download' task when it's instantiated. Fit and evaluate also interface through the `mlcube_utils.py` helpers for reading and writing to disk and calling the appropriate MLCube tasks.
