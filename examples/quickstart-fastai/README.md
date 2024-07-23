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
├── README.md
├── fastai_example
│   ├── client_app.py    # defines your ClientApp
│   └── server_app.py    # defines your ServerApp
└── pyproject.toml       # builds your FAB, includes dependencies and configs
```

### Installing Dependencies

Project dependencies are defined in `pyproject.toml`.
You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually.

### Run with the Simulation Engine

Run:

```bash
flwr run
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config 'num_server_rounds=5'
```

### Alternative way of running the example

You will see that fastai is starting a federated training. For a more in-depth look, be sure to check out the code on our [repo](https://github.com/adap/flower/tree/main/examples/quickstart-fastai).
