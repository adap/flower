---
tags: [quickstart, medical, vision]
dataset: [MedNIST]
framework: [MONAI]
---

# Federated Learning with MONAI and Flower (Quickstart Example)

This introductory example to Flower uses MONAI, but deep knowledge of MONAI is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy. [MONAI](https://docs.monai.io/en/latest/index.html)(Medical Open Network for AI) is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem. This example uses a subset of the [MedMNIST](https://medmnist.com/) dataset including 6 classes, as done in [MONAI's classification demo](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe). This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to partition the data and each client trains am [DenseNet121](https://docs.monai.io/en/stable/networks.html#densenet121) from MONAI. This example runs better with a GPU. If one is detectec, clients will use it.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-monai . \
        && rm -rf _tmp \
        && cd quickstart-monai
```

This will create a new directory called `quickstart-monai` with the following structure:

```shell
quickstart-monai
├── monaiexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `monaiexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5,batch-size=32
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.
