#!/usr/bin/env bash

# TraceFL baseline: Faulty client localisation with label flipping.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_CONFIG="num-server-rounds=3 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=0.7 \
tracefl.max-per-client-data-size=1000 \
tracefl.max-server-data-size=500 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2,3' \
tracefl.faulty-clients-ids='[0]' \
tracefl.label2flip='{1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}'"

echo "Running TraceFL baseline faulty-client experiment..."
flwr run . --run-config "$RUN_CONFIG"

echo "Generating localisation plots for faulty-client scenario..."
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-0.7_rounds-1-2-3_faulty-*.csv" \
  --title "TraceFL Faulty Client Localisation (MNIST)"

