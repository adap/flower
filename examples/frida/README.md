---
tags: [federated-learning, privacy, free-rider-detection, vision, fds]
dataset: [CIFAR-10, CIFAR-100, Fashion-MNIST, Shakespeare]
framework: [torch, torchvision, flwr]
---

# FRIDA: Free-Rider Detection in Federated Learning

FRIDA is a Flower App that implements and evaluates privacy attack-based methods for detecting free-riders in Federated Learning. Free-riders are malicious clients that participate in the federation without contributing meaningful local training, instead sending manipulated or random model updates to benefit from the global model without cost.

FRIDA implements several detection attacks (Cosine, Yeom, DAGMM, L2, STD, DistScore, Inconsistency) and supports multiple free-rider strategies (plain, advanced disguised, gradient noiser) across image and text datasets.

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @pol-garcia-recasens/frida
```

This will create a new directory called `frida` with the following structure:

```shell
frida
├── src/
│   └── setup/
│       ├── __init__.py
│       ├── client_app.py   # Defines your ClientApp
│       ├── server_app.py   # Defines your ServerApp
│       ├── client.py       # Client and free-rider implementations
│       ├── server.py       # Strategy with attack detection
│       ├── attacks.py      # Detection attack implementations
│       ├── model.py        # Model definitions (AlexNet, VGG, LSTM, ...)
│       ├── data_loader.py  # Data loading and partitioning
│       └── train.py        # Training and evaluation logic
├── pyproject.toml          # Project metadata and configuration
└── README.md
```

## Run the App

You can run FRIDA in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend using _simulation_ mode as it requires fewer components to be launched manually.

### Prepare Your Data

FRIDA uses local offline datasets (pickle format). Download and place your dataset under `./data/<dataset_name>/`:

```shell
./data/cifar10/
├── train_data.pkl
└── test_data.pkl
```

### Run with the Simulation Engine

> [!TIP]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to configure CPU/GPU resources for your ClientApp.

Install the dependencies and the `frida` package:

```bash
cd frida && pip install -e .
```

Specify the number of virtual SuperNodes and their resources in `~/.flwr/config.toml`:

```toml
[superlink.local]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 1.0
```

Run with default settings (CIFAR-10, AlexNet, Cosine attack, 10 clients):

```bash
flwr run .
```

You can override settings defined in `pyproject.toml`. For example, to run with a different attack type and number of free-riders:

```bash
flwr run . --run-config "attack_types=yeom num_freeriders=2 num_rounds=10"
```

Key configuration options:

| Parameter        | Description                                                                 | Default   |
| ---------------- | --------------------------------------------------------------------------- | --------- |
| `num_clients`    | Total number of federated clients                                           | `10`      |
| `num_freeriders` | Number of free-rider clients                                                | `0`       |
| `freerider_type` | Free-rider strategy (`none`, `advanced_disguised`, `gradient_noiser`)       | `none`    |
| `attack_types`   | Detection method (`cosine`, `yeom`, `dagmm`, `inconsistency`, `dist_score`) | `cosine`  |
| `dataset`        | Dataset to use (`cifar10`, `cifar100`, `fmnist`, `shakespeare`)             | `cifar10` |
| `architecture`   | Model architecture (`AlexNet`, `VGG19`, `LeNet5`, `LSTM`)                   | `AlexNet` |
| `iid`            | IID or non-IID data partitioning                                            | `true`    |
| `mitigation`     | Exclude detected free-riders from aggregation                               | `false`   |

### Run with the Deployment Engine

In deployment mode, each SuperNode loads its own local data partition from disk.

Prepare one data partition per SuperNode and place it in a local path on each node:

```shell
/path/to/node_data/
├── train_data.pkl
└── test_data.pkl
```

Launch each SuperNode with its local data path:

```shell
flower-supernode \
    --insecure \
    --superlink  \
    --node-config="data_path=/path/to/node_data/"
```

Finally, launch the run via `flwr run` pointing to your SuperLink:

```shell
flwr run .  --stream
```

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run FRIDA with Flower's Deployment Engine. After that, consider setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html).

## Citation

If you use FRIDA in your research, please cite:

```bibtex
@article{recasens2026frida,
  title={Frida: Free-rider detection using privacy attacks},
  author={Recasens, Pol G and Horv{\'a}th, {\'A}d{\'a}m and Gutierrez-Torre, Alberto and Torres, Jordi and Berral, Josep Ll and Pej{\'o}, Bal{\'a}zs},
  journal={Journal of Information Security and Applications},
  volume={97},
  pages={104357},
  year={2026},
  publisher={Elsevier}
}
```
