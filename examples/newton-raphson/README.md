# Flower Example using Distributed Newton-Raphson

The Newton-Raphson strategy is based on Newton-Raphson distributed method. It leads to a faster convergence than `FedAvg`, however it can only be used on convex problems.

See [here](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) for more details.

In this example we will use it in order to train a linear model on this [FLamby dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease) that contains tabular data about patients with heart diseases. Note that the dataset already provides a partitioning of the data for 4 different clients (or 'centers' as they represent medical centers).

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv flower/examples/newton-raphson . && rm -rf _tmp && cd newton-raphson
```

This will create a new directory called `newton-raphson` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

You then need to install specific dependencies for this example. First clone the `FLamby` repo and install it:

```shell
git clone https://github.com/owkin/FLamby.git
poetry run pip install -e "./FLamby[heart]"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

And then you need to clone the `FLamby` repo and install it:

```shell
git clone https://github.com/owkin/FLamby.git
pip install -e "./FLamby[heart]"
```

### Install dataset

In order to install the `FedHeartDisease` dataset, you need to run the following commands:

```shell
cd FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

## Run Federated Learning Example

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
poetry run python main.py
```

Or, if you aren't using `poetry`:

```shell
python3 main.py
```

You should then see the federated training start! Look at the [code](https://github.com/adap/flower/tree/main/examples/newton-raphson) for a detailed explanation.
