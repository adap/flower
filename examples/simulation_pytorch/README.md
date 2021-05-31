# Single-Machine Simulation of Federated Learning Systems

This example is part of a series of blog posts. It's recommended to read the [blog post](https://flower.dev/blog/2021-01-14-single-machine-simulation-of-federated-learning-systems) before reading further.

## Quick Start

If you have docker on your machine you can execute this simulation using it. Start with building the container

```shell
docker build -t flower_federated_learning_simulation_pytorch .
```

and afterwards simply start the simulation in docker using

```shell
docker run --ipc=host -it --rm flower_federated_learning_simulation_pytorch
```

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/simulation_pytorch . && rm -rf flower && cd simulation_pytorch
```

This will create a new directory called `simulation_pytorch` containing the following files:

```shell
-- dataset.py
-- Dockerfile
-- pyproject.toml
-- README.md
-- run.sh
-- SimpleNet.py
-- simulation.py
```

Project dependencies (such as `numpy` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Simulate

To start the simulation simply start the `simulation.py` with Python.

```shell
python3 simulation.py
```

It contains in the last line the following code which you might like to adjust.

```python
if __name__ == "__main__":
    run_simulation(num_rounds=100, num_clients=10, fraction_fit=0.5)
```

If your machine is powerful enough you can try running a single machine simulation with e.g. 100 or even 1000 clients.
