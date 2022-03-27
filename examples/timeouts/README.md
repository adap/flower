# Flower Example demonstrating the usage of request timeouts
This introductory example to Flower demonstrates how request timeouts can be used in Flower.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/timeouts . && rm -rf flower && cd quickstart_tensorflow
```

This will create a new directory called `timeouts` containing the following files:

```shell
-- pyproject.toml
-- client_slow.py
-- client_fast.py
-- server.py
-- README.md
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

If you don't see any errors you're good to go!

# Run the example

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open three more terminals and run the following command in each:

Slow client
```shell
poetry run python3 client_slow.py
```

First fast client 
```shell
poetry run python3 client_fast.py
```

Second fast client 
```shell
poetry run python3 client_fast.py
```

Alternatively you can run all of it in one shell as follows:

```shell
poetry run ./run.sh
```
