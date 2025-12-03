#!/bin/bash

# Recupero variabili da Python
TLS=$(python3 -c "from flwr.common.crypto.config_cripto import TLS; print(TLS)")
NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto import NUM_CLIENTS; print(NUM_CLIENTS)")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ==========================
#  Avvio del superlink con AUTH
# ==========================
if [ "$TLS" = "True" ]; then
    echo "[*] Avvio superlink in modalità TLS + AUTH"
    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --auth-list-public-keys keys/client_public_keys.csv \
        | tee "$LOG_DIR/superlink.log" &
    TLS_FLAG="--root-certificates certificates/ca.crt"
else
    echo "[*] Avvio superlink in modalità INSECURE (auth richiede TLS!)"
    exit 1
fi

sleep 2

# ==========================
#  Avvio dei supernode
# ==========================

CPU_THREADS=6  # Limite thread CPU per client

for ((i=1; i<=NUM_CLIENTS; i++)); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    LOG_FILE="$LOG_DIR/client_$i.log"

    echo "[*] Avvio client $i con max $CPU_THREADS thread CPU"

    OMP_NUM_THREADS=$CPU_THREADS \
    MKL_NUM_THREADS=$CPU_THREADS \
    OPENBLAS_NUM_THREADS=$CPU_THREADS \
    NUMEXPR_NUM_THREADS=$CPU_THREADS \
    VECLIB_MAXIMUM_THREADS=$CPU_THREADS \
    torch_num_threads=$CPU_THREADS \
    flower-supernode \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:${PORT} \
        $TLS_FLAG \
        --auth-supernode-private-key keys/client_credentials_${i} \
        --auth-supernode-public-key keys/client_credentials_${i}.pub \
        --node-config "dataset-path=\"$DATASET\"" \
        | tee "$LOG_FILE" &

    echo "[✓] Avviato client $i su porta $PORT usando keys/client_credentials_${i}"
done

wait
