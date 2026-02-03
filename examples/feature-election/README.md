---
tags: [feature-election, tabular]
dataset: [synthetic]
framework: [sklearn, pandas]
---

# Feature Election: Federated feature selection with Flower

A federated feature selection framework for tabular datasets.

## Overview

Feature Election enables multiple clients with tabular datasets to collaboratively identify the most relevant features without sharing raw data.
This work originates from **FLASH**: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization, presented at [FLTA IEEE 2025](https://flta-conference.org/flta-2025/) (Best Student Paper Award).

**Key Features:**

- **Privacy-preserving**: Clients only share feature selections and scores, not raw data
- **Multiple FS methods**: Lasso, Random Forest, Mutual Information, RFE, and more
- **Configurable aggregation**: Control the balance between intersection and union of features

## Feature Selection Methods

| Method | Description | Speed |
| --------------- | ---------------------------------- | ------ |
| `lasso` | L1-regularized regression (sparse) | Fast |
| `elastic_net` | Elastic Net regularization | Fast |
| `random_forest` | Random Forest importance | Medium |
| `mutual_info` | Mutual information | Medium |
| `f_classif` | F-statistic | Fast |
| `chi2` | Chi-squared test | Fast |
| `rfe` | Recursive Feature Elimination | Slow |
| `pyimpetus` | PyImpetus Markov Blanket | Slow |

## Key Parameters

### Freedom Degree (alpha)

Controls feature selection strategy (alpha in [0,1]):

- **alpha = 0.0**: Intersection - only features selected by ALL clients
- **alpha = 1.0**: Union - features selected by ANY client
- **alpha = 0.5**: Balanced selection (recommended)

### Aggregation Mode

- **weighted**: Weight client contributions by sample count (recommended)
- **uniform**: Equal weight for all clients

## Project Structure

```shell
feature-election/
├── feature_election/              # Package directory
│   ├── __init__.py
│   ├── client_app.py              # ClientApp with @app.train() and @app.evaluate()
│   ├── server_app.py              # ServerApp with @app.main()
│   ├── strategy.py                # Feature Election strategy
│   ├── feature_election_utils.py  # Feature selection methods
│   └── task.py                    # Data loading utilities
├── pyproject.toml                 # Configuration and dependencies
├── README.md
└── test.py                        # Quick verification script
```

## Installation

```bash
# Clone the repository
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/feature-election . && rm -rf flower && feature-election

# Install dependencies
pip install -e .
```

## Running the Project

### Quick Verification

```bash
python test.py
```

### Simulation (CPU)

Run with default parameters:

```bash
flwr run .
```

### Simulation (GPU)

If you have a GPU available:

```bash
flwr run . local-simulation-gpu
```

### Custom Configuration

Override parameters at runtime:

```bash
flwr run . --run-config "freedom-degree=0.3 fs-method='random_forest'"
```

Or edit `pyproject.toml`:

```toml
[tool.flwr.app.config]
freedom-degree = 0.3
aggregation-mode = "weighted"
fs-method = "random_forest"
num-rounds = 1
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Configuration Reference

### Feature Election Parameters

| Parameter | Type | Default | Description |
| ------------------ | ----- | ---------- | -------------------------------------------- |
| `freedom-degree` | float | 0.6 | Selection strategy (0=intersection, 1=union) |
| `aggregation-mode` | str | "weighted" | "weighted" or "uniform" |
| `fs-method` | str | "mutual_info" | Feature selection method |
| `eval-metric` | str | "f1" | Evaluation metric ("f1", "accuracy", "auc") |

### Federated Parameters

| Parameter | Type | Default | Description |
| ------------------- | ----- | ------- | -------------------------------- |
| `num-rounds` | int | 15 | Number of total FL rounds, including tuning and aggreegation |
| `fraction-train` | float | 1.0 | Fraction of nodes for training |
| `fraction-evaluate` | float | 1.0 | Fraction of nodes for evaluation |

## Results

After running, results are saved to `outputs/<date>/<time>/feature_election_results.json`:

```json
{
  "global_feature_mask": [true, false, true, ...],
  "election_stats": {
    "num_clients": 10,
    "num_features_original": 100,
    "num_features_selected": 35,
    "reduction_ratio": 0.65,
    "intersection_features": 15,
    "union_features": 50
  },
  "client_scores": {
    "0": {"initial_score": 0.82, "fs_score": 0.85, "num_features": 30}
  }
}
```

## Algorithm

1. **Client Selection**: Each client performs local feature selection
2. **Score Calculation**: Clients compute feature importance scores
3. **Submission**: Clients send binary masks and scores (not raw data)
4. **Aggregation**: Server aggregates using weighted voting based on `freedom_degree`
5. **Distribution**: Server broadcasts global mask to clients

## Citation

If you use Feature Election in your research, please cite:

FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization

Authors: Ioannis Christofilogiannis, George Valavanis, Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis

FLTA IEEE 2025 Proceedings citation not yet available, please contact: jchr2001@gmail.com

## License

Licensed under the Apache License, Version 2.0.
