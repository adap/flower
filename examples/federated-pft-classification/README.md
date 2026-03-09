---
tags: [advanced, classification, tabular, clinical]
dataset: [PFT (Pulmonary Function Tests)]
framework: [xgboost]
---

# Federated Learning with XGBoost and Flower — PFT Lung Function Classification

The purpose of this project is to use federated learning combined with **PFT** (pulmonary function tests) data pre-processing, to classify patients' lung function into the following categories: mixed restriction and obstruction, obstruction, restriction + small airway obstruction, small airway obstruction, restriction, gas trapping, normal.

We only use spirometry metrics to predict patients' lung function. This way we can sidestep the use of plethysmography machines. This is because spirometry machines are portable and widely available, while plethysmography machines are only found in specialized hospitals. However, plethysmography machines are crucial for the diagnosis of restrictive lung diseases, and with plethysmography results, we can diagnose the patient with standardized decision trees (e.g. **UofT PFT** guidelines). Therefore, we use both spirometry results and plethysmography results to generate the labels using a decision tree, and then train a model to classify lung function based on only spirometry data.

## Data Processing Pipeline

Each client's patient data must contain the following metrics:

| Category | Columns |
|----------|---------|
| Biometrics | `age`, `sex`, `height` |
| Plethysmography | `tlc`, `rv`, `rv_tlc` |
| Spirometry | `fev1`, `fvc`, `fev1_fvc`, `fef75` |

At runtime, the pipeline automatically:
1. Computes LLN (Lower Limit of Normal) and ULN (Upper Limit of Normal) reference values using [GLI 2022](https://doi.org/10.1164/rccm.202205-0963OC) (spirometry) and [ERS 2021](https://doi.org/10.1183/13993003.00289-2020) (lung volumes) equations
2. Labels each patient using the **Computer Aided Decision Tree Used in The Toronto General Pulmonary Function Laboratory**
3. Trains XGBoost using only the spirometry + biometric features (`age`, `height`, `sex`, `fev1`, `fvc`, `fev1_fvc`)

### Label encoding

| Label | Diagnosis |
|-------|-----------|
| 0 | N — normal |
| 1 | AO — obstruction |
| 2 | R — restriction |
| 3 | R+AO — mixed restriction + obstruction |
| 4 | GT — gas trapping |
| 5 | SAO — small airway obstruction |
| 6 | R+SAO — restriction + small airway obstruction |

## Model

We use XGBoost as the primary model because of its superior performance on tabular data, clinical interpretability (feature importance), and its handling of small to medium datasets (datasets are usually of size 1000–8000).

This example provides two federated training strategies:

- **Bagging aggregation** ([`FedXgbBagging`](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedXgbBagging.html)): Each client trains a new tree on its local data each round; all trees are aggregated on the server. With M clients and R rounds, the global model contains M × R trees.
- **Cyclic training** ([`FedXgbCyclic`](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedXgbCyclic.html)): Clients train one at a time in a round-robin fashion, passing the model sequentially.

## Project Structure

```shell
federated-pft-classification
├── federated_pft_classification
│   ├── __init__.py
│   ├── client_app.py          # ClientApp — simulation/deployment dispatch + training
│   ├── server_app.py          # ServerApp — aggregation strategy + model saving
│   ├── task.py                # Data loaders, PFT preprocessing, DMatrix conversion
│   └── data_processing/
│       ├── decision_tree.py   # UofT PFT decision tree
│       ├── gli22_calc.py      # GLI 2022 spirometry reference values
│       ├── ers21_lung_volumes_calc.py  # ERS 2021 lung volume reference values
│       ├── fef75_calc.py      # FEF75 reference values
│       └── main.py            # Combined reference value calculator
├── generate_sim_data.py       # Script to generate a synthetic simulation dataset
├── data/
│   └── simulation_data.xlsx   # Combined patient file used in simulation mode
├── pyproject.toml             # Dependencies and app configuration
└── README.md
```

## Set up the project

### Install dependencies

Create and activate a conda environment, then install the project:

```bash
conda create -n flwr_pft python=3.12 -y
conda activate flwr_pft
pip install -e .
```

### Prepare simulation data

The app expects a single combined Excel file at `data/simulation_data.xlsx` (configurable via `sim-data-path` in `pyproject.toml`). Each row is a patient with the columns listed above.

A synthetic dataset generator is included for testing:

```bash
python generate_sim_data.py
```

This creates `data/simulation_data.xlsx` with 270 synthetic patients across all 7 diagnostic categories.

## Run the project

The app runs smoothly in both **Simulation** and **Deployment** without code changes. If you are starting with Flower, we recommend using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.
> Before you run the flwr run . command, the old Superlink Process might be orphaned and it's input and output file descriptors are made invalid. However, because the network socket might still be intact, it can still accept gRPC requests and fail silently, as such :🎊 Successfully started run 7678037348555877013

```bash

pkill -f "flower-superlink"; flwr run . --stream
```
Note that the simulated data that you created above is mentioned in the pyproject.toml, which will enable the injection of the simulated into context.run_config for every ClientApp call.

You can override settings defined in `pyproject.toml`. For example:

```bash
# Run 10 rounds with cyclic training
flwr run . --run-config "train-method='cyclic' num-server-rounds=10"

# Use a different simulation dataset
flwr run . --run-config "sim-data-path='data/my_dataset.xlsx'"
```

### Run with the Deployment Engine

In deployment, each `SuperNode` loads its own local `.xlsx` file. Pass the path via `node-config`:

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/hospital_data.xlsx"
```

Then launch the run pointing to your `SuperLink`:

```shell
flwr run . <SUPERLINK-CONNECTION> --stream
```

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the app with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

## Configuration reference

Key settings in `pyproject.toml`:

| Key | Default | Description |
|-----|---------|-------------|
| `train-method` | `bagging` | `bagging` or `cyclic` |
| `num-server-rounds` | `3` | Number of federated rounds |
| `local-epochs` | `1` | Trees added per client per round |
| `test-fraction` | `0.2` | Fraction of each client's data held out for validation |
| `sim-data-path` | `data/simulation_data.xlsx` | Combined Excel file for simulation |
| `col-age` … `col-rv-tlc` | `age` … `rv_tlc` | Column names in the Excel files |
| `params.objective` | `multi:softmax` | XGBoost multiclass objective |
| `params.num-class` | `7` | Number of lung function categories |
| `params.eval-metric` | `mlogloss` | Evaluation metric (multiclass log loss) |
