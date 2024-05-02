# Training with Sample-Level Differential Privacy using Opacus Privacy Engine

In this example, we demonstrate how to train a model with differential privacy (DP) using Flower. We employ PyTorch and integrate the Opacus Privacy Engine to achieve sample-level differential privacy. This setup ensures robust privacy guarantees during the client training phase. The code is adapted from the [PyTorch Quickstart example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch).

For more information about DP in Flower please refer to the [tutorial](https://flower.ai/docs/framework/how-to-use-differential-privacy.html). For additional information about Opacus, visit the official \[website\] (https://opacus.ai/).

## Environments Setup

Start by cloning the example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/sample-level-dp-opacus . && rm -rf flower && cd sample-level-dp-opacus
```

This will create a new directory called `sample-level-dp-opacus` containing the following files:

```shell
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing dependencies

Project dependencies are defined in `requirements.txt`. Install them with:

```shell
pip install -r requirements.txt
```

## Run Flower with Opacus and Pytorch

### 1. Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure
```

### 2. Start the long-running Flower clients (SuperNodes)

Start 2 Flower `SuperNodes` in 2 separate terminal windows, using:

```bash
flower-client-app client:app1 --insecure
```

### 3. Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app2 --insecure
```
