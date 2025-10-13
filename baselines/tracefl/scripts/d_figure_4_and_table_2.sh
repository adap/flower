#!/usr/bin/env bash

# TraceFL baseline: Differential privacy ablation.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

NOISES=(0.0001 0.0003 0.0007 0.0009 0.001 0.003)
CLIP=15

for NOISE in "${NOISES[@]}"; do
  RUN_CONFIG="num-server-rounds=2 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=0.3 \
tracefl.max-per-client-data-size=1000 \
tracefl.max-server-data-size=500 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2' \
tracefl.noise-multiplier=${NOISE} \
tracefl.clipping-norm=${CLIP}"

  echo "Running TraceFL baseline with noise=${NOISE} clip=${CLIP}..."
  flwr run . --run-config "$RUN_CONFIG"
done

echo "Generating DP comparison plots..."
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-0.3_rounds-1-2_noise-*_clip-${CLIP}.csv" \
  --title "TraceFL Localization under Differential Privacy"

