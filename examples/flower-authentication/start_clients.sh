#!/bin/bash

NUM_CLIENTS=5

# 1. Genera tutte le chiavi in un colpo solo


# 2. Avvia i client
for i in $(seq 1 $NUM_CLIENTS); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    PRIV_KEY="keys/client_credentials_${i}"
    PUB_KEY="keys/client_credentials_${i}.pub"

    flower-supernode \
      --root-certificates certificates/ca.crt \
      --auth-supernode-private-key "$PRIV_KEY" \
      --auth-supernode-public-key "$PUB_KEY" \
      --node-config "dataset-path=\"$DATASET\"" \
      --clientappio-api-address="0.0.0.0:${PORT}" &

    echo "[✓] Avviato client $i su porta $PORT con dataset $DATASET"
done

wait
