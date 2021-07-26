# Flower Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/sklearn-logreg-mnist . && rm -rf flower && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- utils.py
-- README.md
```

Project dependencies (such as `scikit-learn` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with scikit-learn and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
poetry run python3 client.py
```

Alternatively you can run all of it in one shell as follows:

```shell
poetry run python3 server.py &
poetry run python3 client.py &
poetry run python3 client.py
```

You will see that Flower is starting a federated training. 
