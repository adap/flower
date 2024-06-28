---
title: Sample-Level DP using TensorFlow-Privacy Engine 
labels: [basic, vision, fds, privacy, dp]
dataset: [`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_]
framework: [TensorFlow]
---

# Training with Sample-Level Differential Privacy using TensorFlow-Privacy Engine

In this example, we demonstrate how to train a model with sample-level differential privacy (DP) using Flower. We employ TensorFlow and integrate the tensorflow-privacy Engine to achieve sample-level differential privacy. This setup ensures robust privacy guarantees during the client training phase.

For more information about DP in Flower please refer to the [tutorial](https://flower.ai/docs/framework/how-to-use-differential-privacy.html). For additional information about tensorflow-privacy, visit the official [website](https://www.tensorflow.org/responsible_ai/privacy/guide).

## Environments Setup

Start by cloning the example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/tensorflow-privacy . && rm -rf flower && cd tensorflow-privacy
```

This will create a new directory called `tensorflow-privacy` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- README.md
```

### Installing dependencies

Project dependencies are defined in `pyproject.toml`. Install them with:

```shell
pip install .
```

## Run Flower with tensorflow-privacy and TensorFlow

### 1. Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure
```

### 2. Start the long-running Flower clients (SuperNodes)

Start 2 Flower `SuperNodes` in 2 separate terminal windows, using:

```bash
flower-client-app client:appA --insecure
```

```bash
flower-client-app client:appB --insecure
```

tensorflow-privacy hyperparameters can be passed for each client in `ClientApp` instantiation (in `client.py`). In this example, `noise_multiplier=1.5` and `noise_multiplier=1` are used for the first and second client respectively.

### 3. Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```
