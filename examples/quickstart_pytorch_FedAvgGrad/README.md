# Flower Example using Pytorch with FedAvgGrad strategy

This introductory example to Flower uses PyTorch, but deep knowledge of PyToch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy. Here the focus is given to utilizing the FedAvgGrad strategy. 


## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/danielspg/flower.git && mv flower/examples/quickstart_pytorch_FedAvgGrad . && rm -rf flower && cd quickstart_pytorch_FedAvgGrad
```

This will create a new directory called `quickstart_pytorch` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
```

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with PyTorch and Flower

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
