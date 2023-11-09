# Vertical Federated Learning example

This example will showcase how you can perform Vertical Federated Learning using
Flower. We'll be using the Titanic dataset to train simple regression
models for binary classification.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you
can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/vertical-fl . && rm -rf _tmp && cd vertical-fl
```

This will create a new directory called `vertical-fl` containing the
following files:

```shell
-- pyproject.toml
-- requirements.txt
-- docs/data/train.csv
-- client.py
-- plot.py
-- simulation.py
-- strategy.py
-- task.py
-- README.md
```

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in
`pyproject.toml` and `requirements.txt`. We recommend
[Poetry](https://python-poetry.org/docs/) to install those dependencies and
manage your virtual environment ([Poetry
installation](https://python-poetry.org/docs/#installation)) or
[pip](https://pip.pypa.io/en/latest/development/), but feel free to use a
different way of installing dependencies and managing virtual environments if
you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual
environment. To verify that everything works correctly you can run the following
command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according
to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Usage

Once everything is installed, you can just run:

```shell
poetry run python3 simulation.py
```

for `poetry`, otherwise just run:

```shell
python3 simulation.py
```

This will start the Vertical FL training for 500 rounds with 3 clients.
Eventhough the number of rounds is quite high, this should only take a few
seconds to run as the model is very small.

## Explanations

### Vertical FL

What is Vertical Federated Learning?

### Data

#### Titanic dataset

Context, Features, preprocessing, target, nb samples

#### Partitioning

What do we give to each client?

### Models

What model does the server and clients train on?

### Strategy

What strategy are we using for the aggregation?
