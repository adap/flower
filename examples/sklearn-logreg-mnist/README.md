---
title: Flower LogReg Example using Scikit-Learn
url: https://scikit-learn.org/
labels: [basic, vision, logistic regression, fds]
dataset: [MNIST]
framework: [scikit-learn]
---

# Flower Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/sklearn-logreg-mnist . && rm -rf flower && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- utils.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `scikit-learn` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

## Run Federated Learning with scikit-learn and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two or more terminals and run the following command in each:

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0 # or any integer in {0-9}
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1 # or any integer in {0-9}
```

Alternatively, you can run all of it in one shell as follows:

```bash
bash run.sh
```

You will see that Flower is starting a federated training.
