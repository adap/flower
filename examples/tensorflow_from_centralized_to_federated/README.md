# Tensorflow: From Centralized to Federated Advanced Flower Example (TensorFlow/Keras)

This example demonstrates how an already existing centralized Tensorflow-based machine learning project can be federated with Flower.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/tensorflow_from_centralized_to_federated . && rm -rf flower && cd tensorflow_from_centralized_to_federated
```

This will create a new directory called `tensorflow_from_centralized_to_federated` containing the following files:
```shell
-- client.py
-- poetry.lock
-- pyproject.toml
-- README.md
-- run.sh
-- server.py
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

# Run Federated Learning with TensorFlow/Keras and Flower

The included `run.sh` will start the Flower server (using `server.py`), sleep for 2 seconds to ensure the the server is up, and then start 10 Flower clients (using `client.py`). You can simply start everything in a terminal as follows:

```shell
poetry run ./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anyhting goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).
