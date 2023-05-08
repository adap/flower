# Federated HuggingFace Transformers using Flower and PyTorch

This introductory example to using [HuggingFace](https://huggingface.co) Transformers with Flower with PyTorch. This example has been extended from the [quickstart_pytorch](https://flower.dev/docs/quickstart-pytorch.html) example. The training script closely follows the [HuggingFace course](https://huggingface.co/course/chapter3?fw=pt), so you are encouraged to check that out for detailed explaination for the transformer pipeline.

Like `quickstart_pytorch`, running this example in itself is also meant to be quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_huggingface . && rm -rf flower && cd quickstart_huggingface
```

This will create a new directory called `quickstart_huggingface` containing the following files:

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

# Run Federated Learning with Flower

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

You will see that PyTorch is starting a federated training.
