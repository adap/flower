# Flower Example using Keras

This introductory example to Flower uses Keras but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understanding how to adapt Flower to your use-cases.
Running this example in itself is quite easy.

It is recommended to use [pyenv](https://github.com/pyenv/pyenv)/[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) and [poetry](https://python-poetry.org/docs/) to ensure the right version of libraries.

After installing both tools you can start to set up everything.  

## Setup your virtual environment

First, you should setup your virtualenv and install [Python (version 3.6 or above, we recommend 3.7)](https://docs.python.org/3.7/):

```shell
pyenv install 3.7.9
```

Create a virtualenv with:

```shell
pyenv virtualenv 3.7.9 keras-federated-3.7.9
```

Activate the virtualenv by running the following command:

```shell
echo keras-federated-3.7.9 > .python-version
```

## Run Keras Federated

Start with cloning the Flower repo and checking out the example. We have prepared a single line which you can copy into your shell which will checkout the example for you.

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_tensorflow . && rm -rf flower && cd quickstart_tensorflow
```

You have different files available:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- run-clients.sh
-- run-server.sh
```

The `pyproject.toml` defines the project dependencies. Simply run poetry to install all required dependencies with:

```shell
poetry install
```

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
python3 client.py
```

Alternatively you can run all of it in one shell as follows:

```shell
python3 server.py &
python3 client.py &
python3 client.py &
```

You will see that Keras is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart_tensorflow.html) for a detailed explanation.
