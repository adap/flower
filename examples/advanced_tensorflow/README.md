# Advanced Flower Example (TensorFlow/Keras)

This example demonstrates an advanced federated learning setup using Flower with TensorFlow/Keras. It differs from the quickstart example in the following ways:

- 10 clients (instead of just 2)
- Each client holds a local dataset of 5000 training examples and 1000 test examples
- Server-side model evaluation after parameter aggregation
- Hyperparameter schedule using config functions
- Custom return values
- Server-side parameter initialization

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/advanced_tensorflow . && rm -rf flower && cd advanced_tensorflow
```

This will create a new directory called `advanced_tensorflow` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
-- run.sh
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

## Run Federated Learning with TensorFlow/Keras and Flower

The included `run.sh` will call a script to generate certificates (which will be used by server and clients), start the Flower server (using `server.py`), sleep for 2 seconds to ensure the the server is up, and then start 10 Flower clients (using `client.py`). You can simply start everything in a terminal as follows:

```shell
poetry run ./run.sh
```

The `run.sh` script starts processes in the background so that you don't have to open eleven terminal windows. If you experiment with the code example and something goes wrong, simply using `CTRL + C` on Linux (or `CMD + C` on macOS) wouldn't normally kill all these processes, which is why the script ends with `trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT` and `wait`. This simply allows you to stop the experiment using `CTRL + C` (or `CMD + C`). If you change the script and anything goes wrong you can still use `killall python` (or `killall python3`) to kill all background processes (or a more specific command if you have other Python processes running that you don't want to kill).

## Important / Warning

The approach used to generate SSL certificates can serve as an inspiration and starting point, but it should not be considered as viable for production environments. Please refer to other sources regarding the issue of correctly generating certificates for production environments.
