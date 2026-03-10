---
tags: [quickstart, timeseries]
dataset: [jena-climate]
framework: [tensorflow]
---

# Federated Learning with Tensorflow and Flower

This introductory Flower example uses Tensorflow for timeseries regression, but deep knowledge of Tensorflow is not required to run it. The example is easy to run and helps illustrate how Flower can be adapted to your own use case. It uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition, and preprocess the [Jena-Climate](https://huggingface.co/datasets/sayanroy058/Jena-Climate) dataset.

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @sarah/quickstart-tensorflow-timeseries
```

This will create a new directory called `quickstart-tensorflow-timeseries` with the following structure:

```shell
quickstart-tensorflow-timeseries
tftsexample
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __init__.py
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; client_app.py   # Defines your ClientApp
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; server_app.py   # Defines your ServerApp
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; task.py         # Defines your model, training and data loading
&nbsp;&nbsp;&nbsp;&nbsp; pyproject.toml      # Project metadata like dependencies and configs
&nbsp;&nbsp;&nbsp;&nbsp; README.md
```

## Run the App

You can run your Flower App in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.

Install the dependencies defined in `pyproject.toml` package.

```bash
cd quickstart-tensorflow-timeseries && pip install -e .
```

Run with default settings:

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

### Run with the Deployment Engine

To run this App using Flower's Deployment Engine we recommend first creating some demo data using [Flower Datasets](https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html). For example:

```shell
# Install Flower datasets
pip install "flwr-datasets['vision']"

# Create dataset partitions and save them to disk
flwr-datasets create sayanroy058/Jena-Climate --num-partitions 2 --out-dir demo_data
```

The above command will create two IID partitions of the Jena-climate dataset and save them in a `demo_data` directory. Next, you can pass one partition to each of your `SuperNodes` like this:

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/demo_data/partition_0"
```

Finally, ensure the environment of each `SuperNode` has all dependencies installed.
Then, launch the run via `flwr run` but pointing to a `SuperLink` connection that specifies the `SuperLink` your `SuperNode` is connected to:

```shell
flwr run . <SUPERLINK-CONNECTION> --stream
```

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.