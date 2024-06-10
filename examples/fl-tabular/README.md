# Training with Sample-Level Differential Privacy using Opacus Privacy Engine

Federated Learning on a Tabular Dataset with Flower Framework
This code exemplifies a federated learning setup using the Flower framework on tabular dataset, based on the "Adult Census Income" dataset. The "Adult Census Income" dataset contains demographic information such as age, education, occupation, etc., with the target attribute being income level (<=50K or >50K). The dataset is partitioned into subsets, simulating a federated environment with 5 clients, each holding a distinct portion of the data. Categorical variables are one-hot encoded, and the data is split into training and testing sets. Federated learning is conducted using the FedAvg strategy for 3 rounds.

## Environments Setup

Start by cloning the example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-tabular . && rm -rf flower && cd fl-tabular
```

This will create a new directory called `fl-tabular` containing the following files:

```shell
-- pyproject.toml
-- centralized.py
-- federated.py
-- README.md
```

### Installing dependencies

Project dependencies are defined in `pyproject.toml`. Install them with:

```shell
pip install .
```

## Running Code

### 1. Centralized

```bash
python centralized.py
```

### 2. Federated Using Flower Simulation

```bash
python federated.py
```
