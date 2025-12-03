#!/bin/bash

# ==========================
#  Configurazioni da Python
# ==========================
TLS=$(python3 -c "from flwr.common.crypto.config_cripto import TLS; print(TLS)")
NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto import NUM_CLIENTS; print(NUM_CLIENTS)")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[*] Numero client: $NUM_CLIENTS"
echo "[*] TLS: $TLS"

# ==========================
#  Avvio SuperLink
# ==========================
if [ "$TLS" = "True" ]; then
    echo "[*] Avvio SuperLink in modalità TLS + AUTH"
    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --auth-list-public-keys keys/client_public_keys.csv \
        > "$LOG_DIR/superlink.log" 2>&1 &

    TLS_FLAG="--root-certificates certificates/ca.crt"
    AUTH_ENABLED=true

else
    echo "[*] Avvio SuperLink in modalità INSECURE"
    flower-superlink --insecure > "$LOG_DIR/superlink.log" 2>&1 &

    TLS_FLAG="--insecure"
    AUTH_ENABLED=false
fi

sleep 2

# ==========================
#  Avvio SuperNode
# ==========================
CPU_THREADS=6  # Numero massimo thread CPU per client

for ((i=1; i<=NUM_CLIENTS; i++)); do

    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    LOG_FILE="$LOG_DIR/client_$i.log"

    echo "[*] Avvio client $i su porta $PORT"

    CMD="flower-supernode \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:${PORT} \
        $TLS_FLAG \
        --node-config dataset-path=\"$DATASET\""

    # Aggiungi autenticazione SOLO se TLS è attivo
    if [ "$AUTH_ENABLED" = true ]; then
        CMD="$CMD \
            --auth-supernode-private-key keys/client_credentials_${i} \
            --auth-supernode-public-key keys/client_credentials_${i}.pub"
    fi

    # Avvio processo
    OMP_NUM_THREADS=$CPU_THREADS \
    MKL_NUM_THREADS=$CPU_THREADS \
    OPENBLAS_NUM_THREADS=$CPU_THREADS \
    NUMEXPR_NUM_THREADS=$CPU_THREADS \
    VECLIB_MAXIMUM_THREADS=$CPU_THREADS \
    torch_num_threads=$CPU_THREADS \
    bash -c "$CMD" > "$LOG_FILE" 2>&1 &

    if [ "$AUTH_ENABLED" = true ]; then
        echo "[✓] Client $i AVVIATO (TLS + AUTH), dataset=$DATASET (porta $PORT)"
    else
        echo "[✓] Client $i AVVIATO (INSECURE), dataset=$DATASET (porta $PORT)"
    fi

done

wait
