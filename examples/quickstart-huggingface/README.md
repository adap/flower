---
tags: [quickstart, llm, nlp, sentiment]
dataset: [IMDB]
framework: [transformers]
---

# Federated Learning with HuggingFace Transformers and Flower (Quickstart Example)

This introductory example to using [🤗Transformers](https://huggingface.co/docs/transformers/en/index) with Flower. The training script closely follows the [HuggingFace course](https://huggingface.co/course/chapter3?fw=pt), so you are encouraged to check that out for a detailed explanation of the transformer pipeline.

In this example, we will federated the training of a [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) modle on the [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset. The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/). This example runs best when a GPU is available.

## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/quickstart-huggingface . \
		&& rm -rf _tmp && cd quickstart-huggingface
```

This will create a new directory called `quickstart-huggingface` containing the following files:

```shell
quickstart-huggingface
├── huggingface_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `huggingface_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!TIP\]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below.

```bash
# Run with the default federation (CPU only)
flwr run .
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 1x`ClientApp` (using ~12 GB of VRAM) will run in parallel in each available GPU. Note you can adjust the degree of paralellism but modifying the `client-resources` specification.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config num-server-rounds=5
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart 🤗Transformers tutorial](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower project with the Deployment Engine and TLS certificates, or with Docker.
