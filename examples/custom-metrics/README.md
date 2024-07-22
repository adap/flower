---
title: Example Flower App with Custom Metrics
tags: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [tensorflow]
---

# Flower Example using Custom Metrics

This simple example demonstrates how to calculate custom metrics over multiple clients beyond the traditional ones available in the ML frameworks. In this case, it demonstrates the use of ready-available `scikit-learn` metrics: accuracy, recall, precision, and f1-score.

Once both the test values (`y_test`) and the predictions (`y_pred`) are available on the client side (`client_app.py`), other metrics or custom ones are possible to be calculated.

The main takeaways of this implementation are:

- the use of the `output_dict` on the client side - inside `evaluate` method on `client_app.py`
- the use of the `evaluate_metrics_aggregation_fn` - to aggregate the metrics on the server side, part of the `strategy` on `server_app.py`

This example is based on the `quickstart-tensorflow` with CIFAR-10, source [here](https://flower.ai/docs/quickstart-tensorflow.html), with the addition of [Flower Datasets](https://flower.ai/docs/datasets/index.html) to retrieve the CIFAR-10.

Using the CIFAR-10 dataset for classification, this is a multi-class classification problem, thus some changes on how to calculate the metrics using `average='micro'` and `np.argmax` is required. For binary classification, this is not required. Also, for unsupervised learning tasks, such as using a deep autoencoder, a custom metric based on reconstruction error could be implemented on client side.

## Project Setup

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
              && mv _tmp/examples/custom-metrics . \
              && rm -rf _tmp && cd custom-metrics
```

This will create a new directory called `custom-metrics` containing the
following files:

```shell
custom-metrics
├── README.md
├── custom_metrics_example
│   ├── __init__.py
│   ├── client_app.py    # defines your ClientApp
│   ├── server_app.py    # defines your ServerApp
│   └── task.py
└── pyproject.toml       # builds your FAB, includes dependencies and configs
```

## Install dependencies

```bash
pip install .
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

You will see that Keras is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.ai/docs/quickstart-tensorflow.html) for a detailed explanation. You can add `steps_per_epoch=3` to `model.fit()` if you just want to evaluate that everything works without having to wait for the client-side training to finish (this will save you a lot of time during development).

Running `flwr run` will result in the following output (after 3 rounds):

```shell
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 132.51s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.3027085781097414
INFO :                  round 2: 2.3030176162719727
INFO :                  round 3: 2.307635450363159
INFO :          History (metrics, distributed, evaluate):
INFO :          {'acc': [(1, 0.09419999999999999), (2, 0.1004), (3, 0.0992)],
INFO :           'accuracy': [(1, 0.09419999942183495),
INFO :                        (2, 0.10040000081062317),
INFO :                        (3, 0.09920000061392784)],
INFO :           'f1': [(1, 0.09419999999999999), (2, 0.1004), (3, 0.0992)],
INFO :           'prec': [(1, 0.09419999999999999), (2, 0.1004), (3, 0.0992)],
INFO :           'rec': [(1, 0.09419999999999999), (2, 0.1004), (3, 0.0992)]}
```

### Alternative way of running the example