---
title: Advanced Flower Example using TensorFlow/Keras
url: https://tensorflow.org/
labels: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [TensorFlow, Keras]
---

# Advanced Flower Example (TensorFlow/Keras)

This example demonstrates an advanced federated learning setup using Flower with TensorFlow/Keras. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) and it differs from the quickstart example in the following ways:

- 10 clients (instead of just 2)
- Each client holds a local dataset of 1/10 of the train datasets and 80% is training examples and 20% as test examples (note that by default only a small subset of this data is used when running the `run.sh` script)
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-tensorflow . && rm -rf flower && cd advanced-tensorflow
```

This will create a new directory called `advanced-tensorflow` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
-- run.sh
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

The included `run.sh` will call a script to generate certificates (which will be used by server and clients), start the Flower server (using `server.py`), sleep for 10 seconds to ensure the server is up, and then start 10 Flower clients (using `client.py`). You can simply start everything in a terminal as follows:

```shell
# Once you have activated your environment
./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anything goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

By default `run.sh` uses only a subset of the data. To use the full data, remove the `--toy` argument or set it to False.

## Important / Warning

The approach used to generate SSL certificates can serve as an inspiration and starting point, but it should not be considered as viable for production environments. Please refer to other sources regarding the issue of correctly generating certificates for production environments.
