# Flower Federated Analytics Example

This introductory Flower example demonstrates a Federated Analytics application. It will help you understand how to adapt Flower to your Federated Analytics use cases through a custom strategy. This example uses [Flower Datasets](https://flower.dev/docs/datasets/) to
download, partition and preprocess the dataset.

In this example, we use the Iris dataset splitted between two clients. The subset of each client contains only the features sepal length, and sepal width. Then, a federated analytics task is performed to calculated for each client and each feature its 10-bins histogram, then those values are aggregated and a global histogram is obtained for sepal length, and sepal width.

To learn more about Federated Analytics you can check [this article](https://ai.googleblog.com/2020/05/federated-analytics-collaborative-data.html) by Google. There is also a previous Flower blog post about [this example](https://flower.dev/blog/2023-01-24-federated-analytics-pandas/).

Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
$ git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/federated-analytics . && rm -rf _tmp && cd federated-analytics
```

This will create a new directory called `federated-analytics` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- start.sh
-- README.md
```

### Installing Dependencies

Project dependencies (such as `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Analytics with Flower

After all dependencies installed, you are ready to run this example with the `run.sh` script.

```shell
$ ./run.sh
```

If you don't plan on using the `run.sh` script that automates the run. You can simply start the server in a terminal as follows:

```shell
$ python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
$ python3 client.py --node-id 0
```

Start client 2 in the second terminal:

```shell
$ python3 client.py --node-id 1
```

You will see that the server is printing aggregated statistics about the dataset distributed amongst clients. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart-pandas.html) for a detailed explanation.
