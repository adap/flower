---
tags: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [tensorflow, scikit-learn]
---

# Custom Metrics for Federated Learning with TensorFlow and Flower

This simple example demonstrates how to calculate custom metrics over multiple clients beyond the traditional ones available in the ML frameworks. In this case, it demonstrates the use of ready-available [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html): accuracy, recall, precision, and f1-score.

Once both the test values (`y_test`) and the predictions (`y_pred`) are available on the client side (`client_app.py`), other metrics or custom ones are possible to be calculated.

The main takeaways of this implementation are:

- the return of multiple evaluation metrics generated at the `evaluate` method on `client_app.py`
- the use of the `evaluate_metrics_aggregation_fn` - to aggregate the metrics on the server side, part of the `strategy` on `server_app.py`

This example is based on the `quickstart-tensorflow` with CIFAR-10, source [here](https://flower.ai/docs/quickstart-tensorflow.html), with the addition of [Flower Datasets](https://flower.ai/docs/datasets/index.html) to retrieve the CIFAR-10.

Using the CIFAR-10 dataset for classification, this is a multi-class classification problem, thus some changes on how to calculate the metrics using `average='micro'` and `np.argmax` is required. For binary classification, this is not required. Also, for unsupervised learning tasks, such as using a deep autoencoder, a custom metric based on reconstruction error could be implemented on client side.

## Set up the project

### Clone the project

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
├── custommetrics_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model and dataloading functions
└── pyproject.toml      # Project metadata like dependencies and configs
```

## Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `custommetrics_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
