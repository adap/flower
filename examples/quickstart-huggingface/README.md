---
tags: [quickstart, llm, nlp, sentiment]
dataset: [IMDB]
framework: [transformers]
---

# Federated HuggingFace Transformers using Flower and PyTorch

This introductory example to using [HuggingFace](https://huggingface.co) Transformers with Flower with PyTorch. This example has been extended from the [quickstart-pytorch](https://flower.ai/docs/examples/quickstart-pytorch.html) example. The training script closely follows the [HuggingFace course](https://huggingface.co/course/chapter3?fw=pt), so you are encouraged to check that out for a detailed explanation of the transformer pipeline.

Like `quickstart-pytorch`, running this example in itself is also meant to be quite easy.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-huggingface . \
		&& rm -rf _tmp && cd quickstart-huggingface
```

This will create a new directory called `quickstart-huggingface` containing the following files:

```shell
quickstart-huggingface
├── README.md
├── huggingface_example
│   ├── __init__.py
│   ├── client_app.py
│   └── server_app.py
└── pyproject.toml
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
fewer components to be launched manually. By default, `flwr run` will make
use of the Simluation Engine. Refer to alternative ways of running your
Flower application including Deployment, with TLS certificates, or with
Docker later in this readme.

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
