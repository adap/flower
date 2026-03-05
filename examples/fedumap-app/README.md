# Federated UMAP with Flower

Federated UMAP is a privacy-preserving dimensionality reduction algorithm that produces a 2D UMAP embedding across multiple clients **without any client ever sharing raw data**.

Each client optimises a shared set of landmark points **Y** against its local data using Maximum Mean Discrepancy (MMD) gradient descent. The server aggregates landmark updates via FedAvg, then reconstructs the full pairwise distance matrix using the **Nyström approximation** and runs UMAP on the result.

Two federated embeddings are logged to Weights & Biases (W&B) for comparison:

| Embedding         | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| **Nyström D̂**     | UMAP on the Nyström-reconstructed global distance matrix (Algorithm 2) |
| **K_XY features** | UMAP on Gaussian kernel similarities to the landmarks                  |

## Project Structure

```
federated-umap
├── federated_umap/
│   ├── __init__.py
│   ├── client_app.py   # Defines ClientApp
│   ├── server_app.py   # Defines ServerApp
│   └── task.py         # FedMMDClient, data loading
├── pyproject.toml      # Project metadata and configuration
└── README.md
```

## W&B Setup

Before running, authenticate with W&B:

```bash
wandb login
```

Results are logged to the project defined by `wandb-project` in `pyproject.toml` (default: `federated-umap`). Override at runtime:

```bash
flwr run . --run-config "wandb-project=my-project"
```

## Running the App

### Run with the Simulation Engine

Install the dependencies defined in pyproject.toml as well as the pytorchexample package.

```bash
cd example-app && pip install -e .
```

Run with default settings:

```bash
flwr run .
```

You can also override some of the settings for your ClientApp and ServerApp defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

For more information check Configuration Section

### Run with the Deployment Engine

To run this App using Flower's Deployment Engine we recommend first creating some demo data using Flower Datasets. For example:

Install Flower datasets
```bash
pip install "flwr-datasets['vision']"
```

Create dataset partitions and save them to disk
```bash
flwr-datasets create uoft-cs/cifar10 --num-partitions 2 --out-dir demo_data
```

The above command will create two IID partitions of the CIFAR-10 dataset and save them in a demo_data directory. Next, you can pass one partition to each of your SuperNodes like this:

```bash
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/demo_data/partition_0"
```

Finally, ensure the environment of each SuperNode has all dependencies installed. Then, launch the run via flwr run but pointing to a SuperLink connection that specifies the SuperLink your SuperNode is connected to:

```
flwr run . <SUPERLINK-CONNECTION> --stream
```

## Configuration

All keys live under `[tool.flwr.app.config]` in `pyproject.toml` and can be overridden with `--run-config`.

| Key                 | Default          | Description                                      |
| ------------------- | ---------------- | ------------------------------------------------ |
| `num-server-rounds` | `100`            | Number of federated rounds                       |
| `local-epochs`      | `2`              | Local gradient descent steps per round           |
| `n-y`               | `500`            | Number of UMAP landmark points                   |
| `dataset`           | `ylecun/mnist`   | HuggingFace dataset identifier (Simulation only) |
| `feature-column`    | `image`          | Dataset column containing images                 |
| `label-column`      | `label`          | Dataset column containing class labels           |
| `feature-dim`       | `784`            | Flattened feature dimension (28×28 for MNIST)    |
| `umap-max-samples`  | `10000`          | Max points fed into UMAP at the final round      |
| `wandb-project`     | `federated-umap` | W&B project name                                 |

Specify the number of virtual SuperNodes and their resources in `~/.flwr/config.toml`:

```toml
[superlink.local]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 1.0
```
