#!/bin/bash

# Recupero variabili da Python
TLS=$(python3 -c "from flwr.common.crypto.config_cripto import TLS; print(TLS)")
NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto import NUM_CLIENTS; print(NUM_CLIENTS)")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ==========================
#  Avvio del superlink
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
    AUTH_ENABLED=true

else
    echo "[*] Avvio superlink in modalità INSECURE (no TLS, no AUTH)"

    flower-superlink \
        --insecure \
        | tee "$LOG_DIR/superlink.log" &

    TLS_FLAG="--insecure"
    AUTH_ENABLED=false
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

    # Comando base (comune a TLS e insecure)
    COMMON_CMD="flower-supernode \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:${PORT} \
        $TLS_FLAG \
        --node-config "dataset_path=\"$DATASET\""


    # Se TLS è attivo, aggiungi autenticazione
    if [ "$AUTH_ENABLED" = true ]; then
        COMMON_CMD="$COMMON_CMD \
            --auth-supernode-private-key keys/client_credentials_${i} \
            --auth-supernode-public-key keys/client_credentials_${i}.pub"
    fi

    # Avvio client
    OMP_NUM_THREADS=$CPU_THREADS \
    MKL_NUM_THREADS=$CPU_THREADS \
    OPENBLAS_NUM_THREADS=$CPU_THREADS \
    NUMEXPR_NUM_THREADS=$CPU_THREADS \
    VECLIB_MAXIMUM_THREADS=$CPU_THREADS \
    torch_num_threads=$CPU_THREADS \
    bash -c "$COMMON_CMD" | tee "$LOG_FILE" &

    if [ "$AUTH_ENABLED" = true ]; then
        echo "[✓] Avviato client $i (AUTH ATTIVA) su porta $PORT"
    else
        echo "[✓] Avviato client $i (senza AUTH) su porta $PORT"
    fi
done

wait
