# fl_dp_sa

This is a simple example that utilizes central differential privacy with client-side fixed clipping and secure aggregation.
Note: This example is designed for a small number of rounds and is intended for demonstration purposes.

## Install dependencies

```bash
# Using pip
pip install .

# Or using Poetry
poetry install
```

## Run

The example uses the MNIST dataset with a total of 100 clients, with 20 clients sampled in each round. The hyperparameters for DP and SecAgg are specified in `server.py`.

```shell
flower-simulation --server-app fl_dp_sa.server:app --client-app fl_dp_sa.client:app --num-supernodes 100
```
