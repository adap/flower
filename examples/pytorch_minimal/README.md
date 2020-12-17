# Run PyTorch Centralized and Federated

This PyTorch example is created to show you how to use an already existing centralized machine learning task and run it federated with PyTorch.

This introductory example to Flower uses PyTorch but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you to understand how to adapt Flower to your use-cases.
Running this example in itself is quite easy.

It is recommended to use a virtual environment as described [here](https://flower.dev/docs/recommended-env-setup).
After setting up the virtual environment it is recommended to use [poetry](https://python-poetry.org/docs/) to install the project.

## PyTorch Minimal Files

Start with cloning the Flower repo and checking out the example. We have prepared a single line which you can copy into your shell which will checkout the example for you.

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/src/py/flwr_example/pytorch_minimal . && rm -rf flower && cd pytorch_federated
```

You have different files available:

```shell
-- pyproject.toml
-- cifar.py
-- client.py
-- server.py
-- run-client.sh
-- run-server.sh
```

The `pyproject.toml` defines the project dependencies. Simply run poetry to install all required dependencies with:

```shell
poetry install
```

The other files are described below.

## Centralized and Federated

The PyTorch example uses the CIFAR10 datasets that is a colored image classification task based on the [Deep Learning with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial. The file `cifar.py` contains all the step that are described in the tutorial. It loads the dataset and runs a training based on a CNN model. In addition, the code evaluates the training by measuring loss and accuracy.

You can simply start the centralized training as described in the tutorial by running the `cifar.py`:

```shell
python3 cifar.py
```

If you want to run this pre-existing setup federated you only need `client.py` and `server.py`. The client takes the pre-defined model and training and uses them to setup the Flower workload.

You can simply start first the server in a terminal as follows:

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

The PyTorch example using the CIFAR10 trains now federated.
