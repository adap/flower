# FedHomo 🔐

Federated Learning with Homomorphic Encryption using Flower and TenSEAL.

This example implements the FedAvg strategy with CKKS homomorphic encryption:
model weights are encrypted on the client side before being sent to the server,
which aggregates them without ever seeing plaintext data.

Both **MNIST** and **CIFAR-10** datasets are supported and can be selected
via the `pyproject.toml` configuration.

## Project Structure

```tree
fedhomo/
├── fedhomo/
│ ├── init.py
│ ├── client_app.py
│ ├── client.py
│ ├── encrypted_client.py
│ ├── crypto.py
│ ├── encryption_utils.py
│ ├── dataset.py 
│ ├── model.py 
│ ├── server_app.py
│ ├── strategy.py
│ └── train.py
├── keys/
├── generated_keys.py
├── pyproject.toml
└── README.md
```

## Requirements

Install the dependencies with:

```bash
pip install -e .
```
This will install Flower, TenSEAL, PyTorch and all required dependencies.

## Generate Keys
Before running the example, generate the TenSEAL CKKS keys:

```bash
mkdir -p keys
python generated_keys.py
```

This will populate the ```keys/``` folder with the encryption context and keys
used by clients and server.

## Configuration
In pyproject.toml, you can configure the run:

```text
[tool.flwr.app.config]
dataset = "mnist"        # or "cifar10"
num-server-rounds = 3
num-clients = 2
```
## Run with Flower (Simulation Mode)
```bash
flwr run .
```

## Run with Flower (Deployment Mode)
1. Start the SuperLink:
```bash
flower-superlink --insecure
```

2. Start the SuperNodes:

```bash
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 0.0.0.0:9094 \
    --node-config "partition-id=0 num-partitions=2"

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 0.0.0.0:9095 \
    --node-config "partition-id=1 num-partitions=2"

... and so on for more clients
```

```bash
flwr run . --run-config num-server-rounds=3
```

## Notes

### First Round Decryption Warning
During the **first training round**, clients may log a decryption warning
or error. This is expected behavior: the server aggregates encrypted weights
from scratch and the initial aggregation may produce a ciphertext that is
slightly malformed before the model stabilizes. From the second round onwards
the process runs cleanly.

## How It Works
Key generation — CKKS keys are generated once via generated_keys.py
and stored in the keys/ folder.

Client encryption — each client encrypts its model weights using the
TenSEAL CKKS context before sending them to the server.

Server aggregation — the server aggregates encrypted weights directly
via the custom strategy ```HomomorphicFedAvg``` defined in ```strategy.py```, without decrypting them.

Client decryption — aggregated weights are sent back to clients, which
decrypt them locally and update the model.