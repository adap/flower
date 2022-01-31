# Flower Example using MXNet

This example demonstrates how to run a MXNet machine learning project federated with Flower.

This introductory example for Flower uses MXNet, but you're not required to be a MXNet expert to run the example. The example will help you to understand how Flower can be used to build federated learning use cases based on an existing MXNet projects.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_mxnet . && rm -rf flower && cd quickstart_mxnet
```

This will create a new directory called `quickstart_mxnet` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
```

Project dependencies (such as `mxnet` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Run MXNet Federated

This MXNet example is based on the [Handwritten Digit Recognition](https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/image/mnist.html) tutorial and uses the MNIST dataset (hand-written digits with 28x28 pixels in greyscale with 10 classes). Feel free to consult the tutorial if you want to get a better understanding of MXNet. The file `client.py` contains all the steps that are described in the tutorial. It loads the dataset and a sequential model, trains the model with the training set, and evaluates the trained model on the test set.

You are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

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

You are now training a MXNet-based classifier on MNIST, federated across two clients. The setup is of course simplified since both clients hold the same dataset, but you can now continue with your own explorations. How about changing from a sequential model to a CNN? How about adding more clients?
