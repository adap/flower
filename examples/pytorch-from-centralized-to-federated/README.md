# PyTorch: From Centralized To Federated

This example demonstrates how an already existing centralized PyTorch-based machine learning project can be federated with Flower.

This introductory example for Flower uses PyTorch, but you're not required to be a PyTorch expert to run the example. The example will help you to understand how Flower can be used to build federated learning use cases based on existing machine learning projects. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/pytorch-from-centralized-to-federated . && rm -rf flower && cd pytorch-from-centralized-to-federated
```

This will create a new directory called `pytorch-from-centralized-to-federated` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- cifar.py
-- client.py
-- server.py
-- README.md
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

## From Centralized To Federated

This PyTorch example is based on the [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial and uses the CIFAR-10 dataset (a RGB image classification task). Feel free to consult the tutorial if you want to get a better understanding of PyTorch. The file `cifar.py` contains all the steps that are described in the tutorial. It loads the dataset, trains a convolutional neural network (CNN) on the training set, and evaluates the trained model on the test set.

You can simply start the centralized training as described in the tutorial by running `cifar.py`:

```shell
python3 cifar.py
```

The next step is to use our existing project code in `cifar.py` and build a federated learning system based on it. The only things we need are a simple Flower server (in `server.py`) and a Flower client that connects Flower to our existing model and data (in `client.py`). The Flower client basically takes the already defined model and training code and tells Flower how to call it.

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

You are now training a PyTorch-based CNN image classifier on CIFAR-10, federated across two clients. The setup is of course simplified since both clients hold the same dataset, but you can now continue with your own explorations. How about using different subsets of CIFAR-10 on each client? How about adding more clients?
