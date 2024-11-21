---
tags: [advanced, vision, fds, wandb]
dataset: [Fashion-MNIST]
framework: [keras, tensorflow]
---

# Federated Learning with TensorFlow/Keras and Flower (Advanced Example)

> \[!TIP\]
> This example shows intermediate and advanced functionality of Flower. If you are new to Flower, it is recommended to start from the [quickstart-tensorflow](https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow) example or the [quickstart TensorFlow tutorial](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html).

This example shows how to extend your `ClientApp` and `ServerApp` capabilities compared to what's shown in the [`quickstart-tensorflow`](https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow) example. In particular, it will show how the `ClientApp`'s state (and object of type [RecordSet](https://flower.ai/docs/framework/ref-api/flwr.common.RecordSet.html)) can be used to enable stateful clients, facilitating the design of personalized federated learning strategies, among others. The `ServerApp` in this example makes use of a custom strategy derived from the built-in [FedAvg](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html). In addition, it will also showcase how to:

1. Save model checkpoints
2. Save the metrics available at the strategy (e.g. accuracies, losses)
3. Log training artefacts to [Weights & Biases](https://wandb.ai/site)
4. Implement a simple decaying learning rate schedule across rounds

The structure of this directory is as follows:

```shell
advanced-tensorflow
├── tensorflow_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── strategy.py     # Defines a custom strategy
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

> \[!NOTE\]
> By default this example will log metrics to Weights & Biases. For this, you need to ensure that your system has logged in. Often it's as simple as executing `wandb login` on the terminal after installing `wandb`. Please, refer to this [quickstart guide](https://docs.wandb.ai/quickstart#2-log-in-to-wb) for more information.

This examples uses [Flower Datasets](https://flower.ai/docs/datasets/) with the [Dirichlet Partitioner](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html#flwr_datasets.partitioner.DirichletPartitioner) to partition the [Fashion-MNIST](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) dataset in a non-IID fashion into 50 partitions.

![](_static/fmnist_50_lda.png)

> \[!TIP\]
> You can use Flower Datasets [built-in visualization tools](https://flower.ai/docs/datasets/tutorial-visualize-label-distribution.html) to easily generate plots like the one above.

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorch_example` package. Note that if you want to make use of the GPU, you'll need to install additional packages as described in the [Install Tensorflow](https://www.tensorflow.org/install/pip#linux) documentation.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

When you run the project, the strategy will create a directory structure in the form of `outputs/date/time` and store two `JSON` files: `config.json` containing the `run-config` that the `ServerApp` receives; and `results.json` containing the results (accuracies, losses) that are generated at the strategy.

By default, the metrics: {`centralized_accuracy`, `centralized_loss`, `federated_evaluate_accuracy`, `federated_evaluate_loss`} will be logged to Weights & Biases (they are also stored to the `results.json` previously mentioned). Upon executing `flwr run` you'll see a URL linking to your Weight&Biases dashboard where you can see the metrics.

![](_static/wandb_plots.png)

### Run with the Simulation Engine

With default parameters, 25% of the total 50 nodes (see `num-supernodes` in `pyproject.toml`) will be sampled for `fit` and 50% for an `evaluate` round. By default `ClientApp` objects will run on CPU.

> \[!TIP\]
> To run your `ClientApps` on GPU or to adjust the degree or parallelism of your simulation, edit the `[tool.flwr.federations.local-simulation]` section in the `pyproject.toml`.

```bash
flwr run .

# To disable W&B
flwr run . --run-config use-wandb=false
```

> \[!WARNING\]
> By default TensorFlow processes that use GPU will try to pre-allocate the entire available VRAM. This is undesirable for simulations where we want the GPU to be shared among several `ClientApp` instances. Enable the [GPU memory growth](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth) by setting the `TF_FORCE_GPU_ALLOW_GROWTH` environment variable to ensure processes only make use of the VRAM they need.

You can run the app using another federation (see `pyproject.toml`). For example, if you have a GPU available, select the `local-sim-gpu` federation:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH="true"
flwr run . local-sim-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=10 fraction-fit=0.5"
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.
