---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [tensorflow]
---

# Federated Learning with Tensorflow/Keras and Flower (Quickstart Example)

This introductory example to Flower uses Tensorflow/Keras but deep knowledge of this frameworks is required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-tensorflow . \
        && rm -rf _tmp \
        && cd quickstart-tensorflow
```

This will create a new directory called `quickstart-tensorflow` with the following structure:

```shell
quickstart-tensorflow
├── tfexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `tfhexample` package.

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
flwr run . --run-config num-server-rounds=5,learning-rate=0.05
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart TensorFlow tutorial](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
