# 🌱 FedLer-Farms

Federated Learning for Soil Property Prediction.


## Associated Publication

This Flower application is aligned with:

**Federated earth-observation models for collaborative farm-scale soil mapping**  
*International Journal of Applied Earth Observation and Geoinformation, 146, 105067.*  

<i>Gallios, G., Demattê, J.A.M., Tsakiridis, N., Cardoso, M.C., Kritharoula, A., Tziolas, N. (2026)</i> ([Link](https://doi.org/10.1016/j.jag.2025.105067))


## Overview

Accurate, privacy-preserving soil information is critical for:

- Site-specific nutrient management
- Carbon accounting
- Sustainable agricultural practices


However, laboratory soil analyses remain expensive, resulting in sparse sampling grids at the farm level.

Our project demonstrates a federated learning (FL) framework that enables collaborative soil mapping across distributed farms without sharing raw soil data.

The approach:

1. Uses Sentinel-2 derived features (bare-soil composites)

2. Applies a 1D Convolutional Neural Network (CNN)

3. Trains collaboratively using Federated Averaging (FedAvg)

4. Keeps all local soil data on-premise


## Model Architecture

Implemented in:
```
fedler_farms/model.py
```

**Architecture**:
- 1D Convolutional layer
- ReLU
- 1D Convolutional layer
- ReLU
- Optional max pooling
- Fully connected layer (64 units)
- Multi-output regression head

Designed for spectral or tabular Sentinel-derived features.


## Dataset

Demo dataset hosted on Hugging Face:

```
soil-ai-lab/dummy-soil-dataset
```

The dataset includes:

- Feature columns (X1–X10)
- Targets:

    - Clay_gkg_filtered
    - C_gkg_filtered

<br>
⚠️ Note: This is a simplified demonstration dataset.

The full experimental archive described in the paper is not publicly distributed.

## Installation
```bash
python -m venv fedler-env
source fedler-env/bin/activate
pip install --upgrade pip
pip install -e .
```


## Running (Simulation Engine)

Run fully local federated simulation:

```bash 
flwr run . local-simulation --stream
```

This will:

- Spawn virtual clients
- Partition the dataset
- Train the federated CNN
- Log metrics

## Running (Deployment Engine)

This mode simulates real distributed farms.

### Step 1 — Start SuperLink

```bash
flower-superlink --insecure
```

### Step 2 — Start SuperNodes (Clients)

Example with 3 farms:

```bash
flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9104 
```

```bash
flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9105 
```

```bash
flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9106 
```

### Step 3 — Launch Federated Run

```bash
flwr run . local-deployment --stream
```

Where `local-deployment` is defined in `config.toml`:
```TOML
[superlink.local-deployment]
address = "127.0.0.1:9093"
insecure = true
```

## Metrics

Per round:

- Centralized loss
- Distributed loss
- R² per target
- RMSE per target
- RPIQ per target

Outputs saved to: `outputs/metrics_demo/`.


## Citation

If you use this application, please cite:

```bibtext
@article{Gallios2026FedEO,
  title   = {Federated earth-observation models for collaborative farm-scale soil mapping},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume  = {146},
  pages   = {105067},
  year    = {2026},
  doi     = {10.1016/j.jag.2025.105067},
  author  = {Gallios, G. and Demattê, J.A.M. and Tsakiridis, N. and Cardoso, M.C. and Kritharoula, A. and Tziolas, N.}
}
```

🔗 Link: https://doi.org/10.1016/j.jag.2025.105067

## Developed by

Soil Science Artificial Intelligence Laboratory 

University of Florida
