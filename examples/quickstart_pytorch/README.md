# Flower Example using PyTorch

This introductory example to Flower uses PyTorch, but deep knowledge of PyToch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

It is recommended to use a virtual environment as described [here](https://flower.dev/docs/recommended-env-setup).
After setting up the virtual environment it is recommended to use [poetry](https://python-poetry.org/docs/) to install the project.

## Run PyTorch Federated

Start by cloning the Flower example. We have prepared a single line that you can copy into your shell:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_tensorflow . && rm -rf flower && cd quickstart_tensorflow
```

You have different files available:

```shell
-- pyproject.toml
-- client.py
-- server.py
```

The `pyproject.toml` defines the project dependencies. Simply run poetry to install all required dependencies:

```shell
poetry install
```

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

You will see that PyTorch is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart_pytorch.html) for a detailed explanation.
