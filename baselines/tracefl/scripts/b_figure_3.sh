#!/usr/bin/env bash

# TraceFL baseline: Varying data heterogeneity (Dirichlet alpha sweep).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ALPHAS=(0.1 0.3 0.5 0.7 1.0)

for ALPHA in "${ALPHAS[@]}"; do
  RUN_CONFIG="num-server-rounds=2 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=${ALPHA} \
tracefl.max-per-client-data-size=2048 \
tracefl.max-server-data-size=2048 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2' \
fraction-train=0.4"

  echo "Running TraceFL baseline with Dirichlet alpha=${ALPHA}..."
  flwr run . --run-config "$RUN_CONFIG"
done

echo "Generating combined plots for heterogeneity sweep..."
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-*.csv" \
  --title "TraceFL Localization vs. Data Heterogeneity"

