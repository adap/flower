---
tags: [finetuning, vision, fds]
dataset: [Poultry Health (binary)]
framework: [torch, torchvision]
---

# Federated Finetuning of a Vision Transformer on Poultry Health

This example shows how to use Flower to federate the finetuning of a [ViT-Base-16](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) pretrained on ImageNet. It finetunes just the classification head on a binary poultry health dataset ([Dianyo/poultry-health](https://huggingface.co/datasets/Dianyo/poultry-health)) using [Flower Datasets](https://flower.ai/docs/datasets/) for on-the-fly IID partitioning across 10 clients. Because only the head is trained, each client needs minimal VRAM (~1 GB at batch size 32).

## Set up the project

### Clone the project

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/flowertune-vit-poultry . \
        && rm -rf _tmp \
        && cd flowertune-vit-poultry
### Fetch the app

```shell
flwr new @dianyo/vitpoultry

This will create a new directory called `flowertune-vit-poultry` with the following structure:

```shell
flowertune-vit-poultry
├── vitpoultry
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `vitpoultry` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> **Tip:** Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 batch-size=64"
```

If your system has a GPU you can make use of it:

```bash
flwr run .
```

### Run with the Deployment Engine

To run this app using Flower's Deployment Engine we recommend first creating some demo data using [Flower Datasets](https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html). For example:

```bash
# Install Flower Datasets
pip install "flwr-datasets[vision]"

# Create dataset partitions and save them to disk
flwr-datasets create Dianyo/poultry-health --num-partitions 2 --out-dir demo_data
```

The above command will create two IID partitions of the poultry health dataset and save them in a `demo_data` directory. Next, you can pass one partition to each of your SuperNodes like this:

```bash
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/demo_data/partition_0"
```

Finally, ensure the environment of each SuperNode has all dependencies installed. Then, launch the run via `flwr run` but pointing to a SuperLink connection that specifies the SuperLink your SuperNode is connected to:

```bash
flwr run . <SUPERLINK-CONNECTION> --stream
```

> **Tip:** Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.
