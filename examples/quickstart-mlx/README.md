# Flower Example using MLX

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but deep knowledge of MLX is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon.

In this example, we will train a simple 2 layers MLP on MNIST data (handwritten digits recognition).

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/quickstart-mlx . && rm -rf _tmp && cd quickstart-mlx
```

This will create a new directory called `quickstart-mlx` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `mlx` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with MLX and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminal windows and run the
following commands.

Start a first client in the first terminal:

```shell
python3 client.py
```

And another one in the second terminal:

```shell
python3 client.py
```

If you want to utilize your GPU, you can use the `--gpu` argument:

```shell
python3 client.py --gpu
```

Note that you can start many more clients if you want, but each will have to be in its own terminal.

You will see that MLX is starting a federated training. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-mlx) for a detailed explanation.
