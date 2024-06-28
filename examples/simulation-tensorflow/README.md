---
title: Flower Simulation Example using TensorFlow/Keras
labels: [basic, vision, fds, simulation]
dataset: [`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_]
framework: [TensorFlow, Keras]
---

# Flower Simulation example using TensorFlow/Keras

This introductory example uses the simulation capabilities of Flower to simulate a large number of clients on a single machine. Take a look at the [Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) for a deep dive into how Flower simulation works. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the MNIST dataset. This examples uses 100 clients by default.

## Running the example (via Jupyter Notebook)

Run the example on Google Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/simulation-tensorflow/sim.ipynb)

Alternatively, you can run `sim.ipynb` locally or in any other Jupyter environment.

## Running the example

Start by cloning the code example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/simulation-tensorflow . && rm -rf flower && cd simulation-tensorflow
```

This will create a new directory called `simulation-tensorflow` containing the following files:

```
-- README.md       <- Your're reading this right now
-- sim.ipynb       <- Example notebook
-- sim.py          <- Example code
-- pyproject.toml  <- Example dependencies
-- requirements.txt  <- Example dependencies
```

### Installing Dependencies

Project dependencies (such as `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

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

You can adjust the CPU/GPU resources you assign to each of your virtual clients. By default, your clients will only use 2xCPU core. For example:

```bash
# Will assign 2xCPUs to each client
python sim.py --num_cpus=2

# Will assign 2xCPUs and 25% of the GPU's VRAM to each client
# This means that you can have 4 concurrent clients on each GPU
# (assuming you have enough CPUs)
python sim.py --num_cpus=2 --num_gpus=0.25
```

Because TensorFlow by default maps all the available VRAM, we need to [enable GPU memory growth](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth), see how it is done in the example (`sim.py`) for both the "main" process (where the server/strategy runs) and for the clients (using the `actor_kwargs`)

### Run with Flower Next (preview)

Ensure you have activated your environment, then execute the command below. All `ClientApp` instances will run on CPU but the `ServerApp` will run on the GPU if one is available. Note that this is the case because the `Simulation Engine` only exposes certain resources to the `ClientApp` (based on the `client_resources` in `--backend-config`). For TensorFlow simulations, it is desirable to make use of TF's [memory growth](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth) feature. You can enable that easily with the `--enable-tf-gpu-growth` flag.

```bash
# Run with the default backend-config.
# `--server-app` points to the `server` object in the sim.py file in this example.
# `--client-app` points to the `client` object in the sim.py file in this example.
flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100 --enable-tf-gpu-growth
```

You can change the default resources assigned to each `ClientApp` using the `--backend-config` argument.

```bash
# Tells the VCE to reserve 2x CPUs and 25% of available VRAM for each ClientApp
flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100 \
    --backend-config='{"client_resources": {"num_cpus":2, "num_gpus":0.25}}' --enable-tf-gpu-growth
```

Take a look at the [Documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) for more details on how you can customise your simulation.
