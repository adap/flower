---
title: Example Flower App with Custom Metrics
labels: [basic, vision, fds]
dataset: [CIFAR-10 | https://huggingface.co/datasets/uoft-cs/cifar10]
framework: [tensorflow]
---

# Flower Example using Custom Metrics

This simple example demonstrates how to calculate custom metrics over multiple clients beyond the traditional ones available in the ML frameworks. In this case, it demonstrates the use of ready-available `scikit-learn` metrics: accuracy, recall, precision, and f1-score.

Once both the test values (`y_test`) and the predictions (`y_pred`) are available on the client side (`client.py`), other metrics or custom ones are possible to be calculated.

The main takeaways of this implementation are:

- the use of the `output_dict` on the client side - inside `evaluate` method on `client.py`
- the use of the `evaluate_metrics_aggregation_fn` - to aggregate the metrics on the server side, part of the `strategy` on `server.py`

This example is based on the `quickstart-tensorflow` with CIFAR-10, source [here](https://flower.ai/docs/quickstart-tensorflow.html), with the addition of [Flower Datasets](https://flower.ai/docs/datasets/index.html) to retrieve the CIFAR-10.

Using the CIFAR-10 dataset for classification, this is a multi-class classification problem, thus some changes on how to calculate the metrics using `average='micro'` and `np.argmax` is required. For binary classification, this is not required. Also, for unsupervised learning tasks, such as using a deep autoencoder, a custom metric based on reconstruction error could be implemented on client side.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/custom-metrics . && rm -rf flower && cd custom-metrics
```

This will create a new directory called `custom-metrics` containing the following files:

```shell
-- pyproject.toml
-- requirements.txt
-- client.py
-- server.py
-- run.sh
-- README.md
```

### Installing Dependencies

Project dependencies (such as `scikit-learn`, `tensorflow` and `flwr`) are defined in `pyproject.toml` and `requirements.txt`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)) or [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

#### Poetry

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Federated Learning with Custom Metrics

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
python client.py
```

Alternatively you can run all of it in one shell as follows:

```shell
python server.py &
# Wait for a few seconds to give the server enough time to start, then:
python client.py &
python client.py
```

or

```shell
chmod +x run.sh
./run.sh
```

You will see that Keras is starting a federated training. Have a look to the [Flower Quickstarter documentation](https://flower.ai/docs/quickstart-tensorflow.html) for a detailed explanation. You can add `steps_per_epoch=3` to `model.fit()` if you just want to evaluate that everything works without having to wait for the client-side training to finish (this will save you a lot of time during development).

Running `run.sh` will result in the following output (after 3 rounds):

```shell
INFO flwr 2024-01-17 17:45:23,794 | app.py:228 | app_fit: metrics_distributed {
    'accuracy': [(1, 0.10000000149011612), (2, 0.10000000149011612), (3, 0.3393000066280365)],
    'acc': [(1, 0.1), (2, 0.1), (3, 0.3393)],
    'rec': [(1, 0.1), (2, 0.1), (3, 0.3393)],
    'prec': [(1, 0.1), (2, 0.1), (3, 0.3393)],
    'f1': [(1, 0.10000000000000002), (2, 0.10000000000000002), (3, 0.3393)]
}
```
