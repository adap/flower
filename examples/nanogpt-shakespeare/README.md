______________________________________________________________________

## tags: [basic, nlp, gpu] dataset: [Tiny Shakespeare] framework: [torch]

# Federated Learning with NanoGPT and Flower

This example demonstrates federated learning for character-level language
modeling using [NanoGPT](https://github.com/karpathy/nanoGPT) and Flower.
Multiple clients each train a small GPT model on their own partition of the
[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
dataset, and the server aggregates weights using FedAvg. After training, the
server generates a Shakespeare-style text sample from the final model.

The model is a "baby GPT" (~10.8M parameters) — small enough to train on CPU.

## Fetch the App

Install Flower and clone the app:

```shell
pip install "flwr[simulation]>=1.26.1"
flwr new @kungfuai/nanogpt-shakespeare --framework PyTorch
```

This will create the following project layout:

```shell
nanogpt-shakespeare
├── nanogpt_shakespeare
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # NanoGPT model, data loading, train/test
├── pyproject.toml       # Project metadata and Flower config
└── README.md
```

Install the dependencies:

```bash
pip install -e .
```

## Run the App

### Run with the Simulation Engine

> [!TIP]
> Learn more about Flower's Simulation Engine in the
> [documentation](https://flower.ai/docs/framework/how-to-run-simulations.html).

```bash
flwr run .
```

You can override configuration values:

```bash
flwr run . --run-config "num-server-rounds=10 learning-rate=5e-4 max-steps=100"
```

Expected output over 5 rounds (2 simulated clients, 50 steps each, CPU):

| Round | Train Loss | Val Loss | Perplexity |
|-------|-----------|----------|------------|
| 0 | — | 4.23 | 69.0 |
| 1 | 3.26 | 2.97 | 19.5 |
| 2 | 2.74 | 2.64 | 14.0 |
| 3 | 2.60 | 2.56 | 13.0 |
| 4 | 2.53 | 2.52 | 12.5 |
| 5 | 2.50 | 2.52 | 12.4 |

### Run with the Deployment Engine

> [!TIP]
> Follow the [Deployment Engine guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
> to learn about deploying Flower apps. For production use, consider
> [enabling TLS](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)
> and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html).

First, prepare demo data partitions. The dataset is Tiny Shakespeare (~1MB)
which is downloaded automatically. To create partitions for deployment, run:

```bash
python -c "
from nanogpt_shakespeare.task import _download_shakespeare, _get_meta, _encode
import numpy as np
from pathlib import Path

text = _download_shakespeare()
meta = _get_meta()
ids = np.array(_encode(text, meta['stoi']), dtype=np.uint16)
n = len(ids)
split = int(n * 0.9)
train_ids, val_ids = ids[:split], ids[split:]

for i, num in enumerate([2]):
    chunk = len(train_ids) // num
    for p in range(num):
        out = Path(f'datasets/shakespeare_part_{p}')
        out.mkdir(parents=True, exist_ok=True)
        start = p * chunk
        end = start + chunk if p < num - 1 else len(train_ids)
        train_ids[start:end].tofile(out / 'train.bin')
        val_ids.tofile(out / 'val.bin')
        print(f'Created partition {p} at {out}')
"
```

Then launch SuperNodes, each pointing to a data partition:

```shell
flower-supernode --insecure --superlink="SUPERLINK_IP:9092" \
                 --node-config="data-path='datasets/shakespeare_part_0'"
```

Finally, run the app in your federation:

```shell
flwr run . my-federation
```

## Configuration

| Parameter | Default | Description |
|---------------------|---------|--------------------------------------|
| num-server-rounds | 5 | Number of federated rounds |
| local-epochs | 1 | Training epochs per client per round |
| learning-rate | 1e-3 | AdamW learning rate |
| batch-size | 64 | Training batch size |
| block-size | 256 | Context window length (chars) |
| max-steps | 50 | Max training steps per round (0 = unlimited) |
| n-layer | 6 | Transformer layers |
| n-head | 6 | Attention heads |
| n-embd | 384 | Embedding dimension |
| dropout | 0.2 | Dropout rate |
