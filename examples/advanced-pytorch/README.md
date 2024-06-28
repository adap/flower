---
title: Advanced Flower Example using PyTorch
labels: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Advanced Flower Example (PyTorch)

This example demonstrates an advanced federated learning setup using Flower with PyTorch. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) and it differs from the quickstart example in the following ways:

- 10 clients (instead of just 2)
- Each client holds a local dataset of 5000 training examples and 1000 test examples (note that using the `run.sh` script will only select 10 data samples by default, as the `--toy` argument is set).
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced-pytorch . && rm -rf flower && cd advanced-pytorch
```

This will create a new directory called `advanced-pytorch` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
-- run.sh
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Learning with PyTorch and Flower

The included `run.sh` will start the Flower server (using `server.py`),
sleep for 2 seconds to ensure that the server is up, and then start 10 Flower clients (using `client.py`) with only a small subset of the data (in order to run on any machine),
but this can be changed by removing the `--toy` argument in the script. You can simply start everything in a terminal as follows:

```shell
# After activating your environment
./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anything goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

You can also manually run `python3 server.py` and `python3 client.py --client-id <ID>` for as many clients as you want but you have to make sure that each command is run in a different terminal window (or a different computer on the network). In addition, you can make your clients use either `EfficienNet` (default) or `AlexNet` (but all clients in the experiment should use the same). Switch between models using the `--model` flag when launching `client.py` and `server.py`.
