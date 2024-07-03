---
tags: [quickstart, tabular, federated analytics]
dataset: [Iris]
framework: [pandas]
---

# Flower Example using Pandas

This introductory example to Flower uses Pandas, but deep knowledge of Pandas is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to
download, partition and preprocess the dataset.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
$ git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart-pandas . && rm -rf _tmp && cd quickstart-pandas
```

This will create a new directory called `quickstart-pandas` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- start.sh
-- README.md
```

If you don't plan on using the `run.sh` script that automates the run, you should first download the data and put it in a `data` folder, this can be done by executing:

```shell
$ mkdir -p ./data
$ python -c "from sklearn.datasets import load_iris; load_iris(as_frame=True)['data'].to_csv('./data/client.csv')"
```

### Installing Dependencies

Project dependencies (such as `pandas` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Analytics with Pandas and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
$ python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
$ python3 client.py --partition-id 0
```

Start client 2 in the second terminal:

```shell
$ python3 client.py --partition-id 1
```

You will see that the server is printing aggregated statistics about the dataset distributed amongst clients. Have a look to the [Flower Quickstarter documentation](https://flower.ai/docs/quickstart-pandas.html) for a detailed explanation.
