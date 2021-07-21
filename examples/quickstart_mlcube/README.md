# Flower Example using TensorFlow/Keras + MLCube

This introductory example to Flower uses MLCube together with Keras but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understanding how to adapt Flower to your use-cases with MLCube.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_mlcube . && rm -rf flower && cd quickstart_mlcube
```

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

For the MLCube setup you will need to install Docker on your system. Please refer to the [Docker install guide](https://docs.docker.com/get-docker/) on how to do that.

If you don't see any errors you're good to go!

# Run Federated Learning with TensorFlow/Keras in MLCube with Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python3 server.py
```

Now you are ready to start everything. We have prepared a simple script in the example root called `run.sh` which you can execute like this:

```shell
./run.sh
```

To understand whats happening in the script please checkout its [content](https://github.com/adap/flower/blob/main/examples/quickstart_mlcube/run.sh).

Congrats! You have just run a Federated Learning experiment using TensorFlow/Keras in MLCube using Flower for federation.
