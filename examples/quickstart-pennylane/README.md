---
tags: [quantum, vision, fds]
dataset: [CIFAR-10]
framework: [pennylane, torch, torchvision]
---

# Quantum Federated Learning with PennyLane and Flower

This tutorial demonstrates how to implement **Quantum Federated Learning** using [PennyLane](https://pennylane.ai/) for a hybrid quantum-classical neural network and [Flower](https://flower.ai/) for federated learning. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset for training a hybrid quantum-classical neural network.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pennylane . \
        && rm -rf _tmp \
        && cd quickstart-pennylane
```

This will create a new directory called `quickstart-pennylane` with the following structure:

```shell
quickstart-pennylane
├── quickstart_pennylane
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `quickstart_pennylane` package.

```bash
pip install -e .
```

## Run the Project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example might run slower due to quantum circuit simulation. If your system has a GPU, you can use the `local-simulation-gpu` federation for better performance. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.01"
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 5x`ClientApp` will run in parallel in the available GPU. You can tweak the degree of parallelism by adjusting the settings of this federation in the `pyproject.toml`.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
