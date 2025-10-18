#!/usr/bin/env bash

# TraceFL baseline: Localization accuracy on correctly classified samples.
# This mirrors Figure 2 / Table 3 / Figure 5 from the original repository.

set -euo pipefail

# Always run from the TraceFL baseline repository root regardless of the
# invocation directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_CONFIG="num-server-rounds=2 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=0.3 \
tracefl.max-per-client-data-size=2048 \
tracefl.max-server-data-size=2048 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2' \
tracefl.use-deterministic-sampling=true \
tracefl.random-seed=42 \
min-train-nodes=4"

echo "Running TraceFL baseline localization experiment..."
flwr run . --run-config "$RUN_CONFIG"

echo "Generating accuracy plots..."
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-0-3_rounds-1-2*.csv" \
  --title "TraceFL Localization Accuracy (MNIST, 10 Clients)"

