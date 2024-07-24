---
tags: [quickstart, vision]
dataset: [MNIST]
framework: [fastai]
---

# Flower Example using fastai

This introductory example to Flower uses [fastai](https://www.fast.ai/), but deep knowledge of fastai is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-fastai . \
		&& rm -rf _tmp && cd quickstart-fastai
```

This will create a new directory called `quickstart-fastai` containing the following files:

```shell
quickstart-fastai
├── README.md
├── fastai_example
│   ├── client_app.py    # defines your ClientApp
│   └── server_app.py    # defines your ServerApp
└── pyproject.toml       # builds your project, includes dependencies and configs
```

## Install dependencies

Install the dependencies defined in `pyproject.toml` as well as the `fastai_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make
use of the Simluation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You will see that fastai is starting a federated training. For a more in-depth look, be sure to check out the code on our [repo](https://github.com/adap/flower/tree/main/examples/quickstart-fastai).

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
