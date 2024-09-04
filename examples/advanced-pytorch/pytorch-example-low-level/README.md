# Federated Learning with PyTorch and Flower (Advanced Example -- Low level API)

> \[!WARNING\]
> This example uses Flower's low-level API which is in a preview state subject to change.

```shell
pytorch-pytorch-low-level
├── pytorch_example_low_level
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── utils.py        # Defines utility functions
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorch_example_low_level` package.

```bash
pip install -e .
```

## Run the project

### Run with the Simulation Engine

With default parameters, 20% of the total 50 nodes (see `num-supernodes` in `pyproject.toml`) will be sampled in each round. By default `ClientApp` objects will run on CPU.

```bash
flwr run .
```
