---
tags: [basic, vision, fds, simulation]
dataset: [MNIST]
framework: [torch, torchvision]
---

# Flower Simulation example using PyTorch

This introductory example uses the simulation capabilities of Flower to simulate a large number of clients on a single machine. Take a look at the [Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) for a deep dive into how Flower simulation works. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset. This examples uses 100 clients by default.

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
poetry run python -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

### Run with `start_simulation()`

Ensure you have activated your environment then:

```bash
# and then run the example
python sim.py
```

You can adjust the CPU/GPU resources you assign to each of your virtual clients. By default, your clients will only use 1xCPU core. For example:

```bash
# Will assign 2xCPUs to each client
python sim.py --num_cpus=2

# Will assign 2xCPUs and 25% of the GPU's VRAM to each client
# This means that you can have 4 concurrent clients on each GPU
# (assuming you have enough CPUs)
python sim.py --num_cpus=2 --num_gpus=0.25
```

### Run with Flower Next (preview)

Ensure you have activated your environment, then execute the command below. All `ClientApp` instances will run on CPU but the `ServerApp` will run on the GPU if one is available. Note that this is the case because the `Simulation Engine` only exposes certain resources to the `ClientApp` (based on the `client_resources` in `--backend-config`).

```bash
# Run with the default backend-config.
# `--server-app` points to the `server` object in the sim.py file in this example.
# `--client-app` points to the `client` object in the sim.py file in this example.
flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100
```

You can change the default resources assigned to each `ClientApp` by means of the `--backend-config` argument:

```bash
# Tells the VCE to reserve 2x CPUs and 25% of available VRAM for each ClientApp
flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100 \
    --backend-config='{"client_resources": {"num_cpus":2, "num_gpus":0.25}}'
```

Take a look at the [Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) for more details on how you can customise your simulation.
