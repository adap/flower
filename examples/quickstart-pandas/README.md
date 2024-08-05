---
tags: [quickstart, tabular, federated analytics]
dataset: [Iris]
framework: [pandas]
---

# Federated Learning with Pandas and Flower (Quickstart Example)

This introductory example to Flower uses Pandas, but deep knowledge of Pandas is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to
download, partition and preprocess the [Iris dataset](https://huggingface.co/datasets/scikit-learn/iris).
Running this example in itself is quite easy.

This example implements a form of Federated Analyics by which nodes, instead of training a model using locally available data, they run a quiery on the data they own. In this example the query is to compute the histogram on specific columns of the dataset. These metrics are sent to the server for aggregation.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-pandas . \
		&& rm -rf _tmp && cd quickstart-pandas
```

This will create a new directory called `quickstart-pandas` containing the following files:

```shell
quickstart-pandas
├── README.md
├── pandas_example
│   ├── client_app.py    # defines your ClientApp
│   └── server_app.py    # defines your ServerApp
└── pyproject.toml       # builds your project, includes dependencies and configs
```

## Install dependencies

Install the dependencies defined in `pyproject.toml` as well as the `pandas_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . --run-config 'num_server_rounds=5'
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart Pandas tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pandas.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
