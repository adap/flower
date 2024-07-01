---
title: Flower Example on Adult Census Income Tabular Dataset
labels: [basic, tabular, fds]
dataset: [Adult Census Income | https://www.kaggle.com/datasets/uciml/adult-census-income/data]
framework: [scikit-learn | https://scikit-learn.org/, torch | https://pytorch.org/]
---

# Flower Example on Adult Census Income Tabular Dataset

This code exemplifies a federated learning setup using the Flower framework on the ["Adult Census Income"](https://huggingface.co/datasets/scikit-learn/adult-census-income) tabular dataset. The "Adult Census Income" dataset contains demographic information such as age, education, occupation, etc., with the target attribute being income level (\<=50K or >50K). The dataset is partitioned into subsets, simulating a federated environment with 5 clients, each holding a distinct portion of the data. Categorical variables are one-hot encoded, and the data is split into training and testing sets. Federated learning is conducted using the FedAvg strategy for 5 rounds.

This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the dataset.

## Environments Setup

Start by cloning the example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-tabular . && rm -rf flower && cd fl-tabular
```

This will create a new directory called `fl-tabular` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- task.py
-- README.md
```

### Installing dependencies

Project dependencies are defined in `pyproject.toml`. Install them with:

```shell
pip install .
```

## Running Code

### Federated Using Flower Simulation

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 5
```
