#!/bin/bash

# ============================================================
#   LETTURA CONFIG DA PYTHON
# ============================================================

TLS=$(python3 -c "from flwr.common.crypto.config_cripto import TLS; print(TLS)")
NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto import NUM_CLIENTS; print(NUM_CLIENTS)")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[*] Avvio con TLS=$TLS, NUM_CLIENTS=$NUM_CLIENTS"


# ============================================================
#   AVVIO SUPERLINK
# ============================================================

if [ "$TLS" = "True" ]; then
    echo "[*] Avvio superlink in modalità TLS"
    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        2>&1 | tee "$LOG_DIR/superlink.log" &
else
    echo "[*] Avvio superlink in modalità INSECURE"
    flower-superlink \
        --insecure \
        2>&1 | tee "$LOG_DIR/superlink.log" &
fi

sleep 2


# ============================================================
#   LIMITAZIONE RISORSE CPU PER I CLIENT
# ============================================================

CPU_THREADS=6
echo "[*] Ogni client userà massimo $CPU_THREADS thread CPU"


# ============================================================
#   AVVIO CLIENT (SUPERNODE)
# ============================================================

for ((i=1; i<=NUM_CLIENTS; i++)); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    LOG_FILE="$LOG_DIR/client_$i.log"

    echo "[*] Avvio client $i su porta $PORT con dataset $DATASET"

    export OMP_NUM_THREADS=$CPU_THREADS
    export MKL_NUM_THREADS=$CPU_THREADS
    export OPENBLAS_NUM_THREADS=$CPU_THREADS
    export NUMEXPR_NUM_THREADS=$CPU_THREADS
    export VECLIB_MAXIMUM_THREADS=$CPU_THREADS
    export torch_num_threads=$CPU_THREADS

    if [ "$TLS" = "True" ]; then
        flower-supernode \
            --superlink 127.0.0.1:9092 \
            --root-certificates certificates/ca.crt \
            --auth-supernode-private-key keys/client_credentials_$i \
            --auth-supernode-public-key keys/client_credentials_$i.pub \
            --node-config "dataset-path='$DATASET' fit_timeout=2000 evaluate_timeout=120 ray_timeout=600 connect_timeout=60" \
            --clientappio-api-address 0.0.0.0:${PORT} \
            2>&1 | tee "$LOG_FILE" &
    else
        flower-supernode \
            --superlink 127.0.0.1:9092 \
            --insecure \
            --node-config "dataset-path='$DATASET' fit_timeout=2000 evaluate_timeout=120 ray_timeout=600 connect_timeout=60" \
            --clientappio-api-address 0.0.0.0:${PORT} \
            2>&1 | tee "$LOG_FILE" &
    fi

    echo "[✓] Client $i avviato (log: $LOG_FILE)"
done


# ============================================================
#   ATTESA TERMINAZIONE
# ============================================================

wait
