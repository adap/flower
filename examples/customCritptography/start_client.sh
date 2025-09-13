#!/bin/bash

NUM_CLIENTS=20
LOG_DIR="logs"

mkdir -p $LOG_DIR

for ((i=1; i<=NUM_CLIENTS; i++)); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    LOG_FILE="$LOG_DIR/client_$i.log"

    flower-supernode \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 0.0.0.0:${PORT} \
      --insecure \
      --node-config "dataset-path=\"$DATASET\"" \
      >"$LOG_FILE" 2>&1 &

    echo "[✓] Avviato client $i su porta $PORT con dataset $DATASET (log: $LOG_FILE)"
done

# Attendi la terminazione di tutti i processi
wait
