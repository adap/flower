# Flower Simulation example using PyTorch

This introductory example uses the simulation capabilities of Flower to simulate a large number of clients on either a single machine or a cluster of machines.

## Running the example (via Jupyter Notebook)

Run the example on Google Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/simulation-pytorch/sim.ipynb)

Alternatively, you can run `sim.ipynb` locally or in any other Jupyter environment.

## Running the example

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/simulation-pytorch . && rm -rf flower && cd simulation-pytorch
```

This will create a new directory called `simulation-pytorch` containing the following files:

```
-- README.md         <- Your're reading this right now
-- sim.ipynb         <- Example notebook
-- sim.py            <- Example code
-- utils.py          <- auxiliary functions for this example
-- pyproject.toml    <- Example dependencies
-- requirements.txt  <- Example dependencies
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

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

### Run Federated Learning Example

```bash
# You can run the example without activating your environemnt
poetry run python3 sim.py

# Or by first activating it
poetry shell
# and then run the example
python sim.py
# you can exit your environment by typing "exit"
```
