---
tags: [DP, DP-SGD, basic, vision, fds, privacy]
dataset: [MNIST]
framework: [tensorflow]
---

# Training with Sample-Level Differential Privacy using TensorFlow-Privacy Engine

In this example, we demonstrate how to train a model with sample-level differential privacy (DP) using Flower. We employ TensorFlow and integrate the tensorflow-privacy engine to achieve sample-level differential privacy. This setup ensures robust privacy guarantees during the client training phase.

For more information about DP in Flower please refer to the [tutorial](https://flower.ai/docs/framework/how-to-use-differential-privacy.html). For additional information about tensorflow-privacy, visit the official [website](https://www.tensorflow.org/responsible_ai/privacy/guide).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git \
        && mv flower/examples/tensorflow-privacy . \
        && rm -rf flower \
        && cd tensorflow-privacy
```

This will create a new directory called `tensorflow-privacy` containing the following files:

```shell
tensorflow-privacy
├── tf_privacy
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

> [!NOTE]
> Please note that, at the current state, users cannot set `NodeConfig` for simulated `ClientApp`s. For this reason, the hyperparameter `noise_multiplier` is set in the `client_fn` method based on a condition check on `partition_id`. This will be modified in a future version of Flower to allow users to set `NodeConfig` for simulated `ClientApp`s.

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `tf_privacy` package.

```shell
# From a new python environment, run:
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
flwr run . --run-config "l2-norm-clip=1.5 num-server-rounds=5"
```
