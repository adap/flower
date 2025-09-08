#!/bin/bash

NUM_CLIENTS=5

# Avvia i client in insecure mode (senza certificati e auth)
for i in $(seq 1 $NUM_CLIENTS); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"

    flower-supernode \
      --insecure \
      --node-config "dataset-path=\"$DATASET\"" \
      --clientappio-api-address="0.0.0.0:${PORT}" &

    echo "[✓] Avviato client $i su porta $PORT con dataset $DATASET (insecure mode)"
done

wait
