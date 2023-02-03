# Flower Example using fastai

This introductory example to Flower uses [fastai](https://www.fast.ai/), but deep knowledge of fastai is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart_fastai . && rm -rf _tmp && cd quickstart_fastai
```

This will create a new directory called `quickstart_fastai` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- run.sh
-- README.md
```

Project dependencies (such as `fastai` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with fastai and Flower

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

You will see that fastai is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.dev/docs/quickstart-fastai.html) for a detailed explanation.
