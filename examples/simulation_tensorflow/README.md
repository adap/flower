# EXPERIMENTAL Flower Simulation Example using TensorFlow/Keras

This introductory example uses the simulation capabilities of Flower to simulate a large number of clients on either a single machine of a cluster of machines.

## Running the example (via Jupyter Notebook)

Run the example on Google Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/quickstart_simulation/sim.ipynb)

Alternatively, you can run `sim.ipynb` locally or in any other Jupyter environment.

## Running the example (via Poetry)

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/quickstart_simulation . && rm -rf flower && cd quickstart_simulation
```

This will create a new directory called `quickstart_simulation` containing the following files:

```shell
-- README.md       <- Your're reading this right now
-- sim.ipynb       <- Example notebook
-- sim.py          <- Example code
-- pyproject.toml  <- Example dependencies (for Poetry)
```

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` (the modern alternative to `requirements.txt`). We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go! 

```bash
poetry run python3 sim.py
```
