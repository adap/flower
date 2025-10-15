#!/usr/bin/env bash

# TraceFL baseline: Differential Privacy (Figure-4 and Table-2)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running TraceFL baseline differential privacy experiment..."

RUN_CONFIG="num-server-rounds=2 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=0.3 \
tracefl.max-per-client-data-size=2048 \
tracefl.max-server-data-size=2048 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2' \
tracefl.noise-multiplier=0.001 \
tracefl.clipping-norm=15 \
fraction-train=0.4"

echo "Running TraceFL baseline with differential privacy..."
flwr run . --run-config "$RUN_CONFIG"

echo "Generating plots for differential privacy analysis..."
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-0.3*.csv" \
  --title "TraceFL with Differential Privacy"
