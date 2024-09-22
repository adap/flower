---
tags: [quickstart, vision]
dataset: [MNIST]
framework: [fastai]
---

# Federated Learning with fastai and Flower (Quickstart Example)

This introductory example to Flower uses [fastai](https://www.fast.ai/), but deep knowledge of fastai is not necessarily required to run the example. The example will help you understand how to adapt Flower to your specific use case, and running it is quite straightforward.

fastai is a deep learning library built on PyTorch which provides practitioners with high-level components for building deep learning projects. In this example, we will train a [SqueezeNet v1.1](https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1) model on the [MNIST](https://huggingface.co/datasets/ylecun/mnist) dataset. The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-fastai . \
		&& rm -rf _tmp && cd quickstart-fastai
```

This will create a new directory called `quickstart-fastai` containing the following files:

```shell
quickstart-fastai
├── fastai_example
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fastai_example` package.

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
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
