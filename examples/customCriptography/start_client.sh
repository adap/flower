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
    echo "[*] Avvio superlink in modalità TLS"
    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        | tee "$LOG_DIR/superlink.log" &
    TLS_FLAG="--root-certificates certificates/ca.crt"
else
    echo "[*] Avvio superlink in modalità INSECURE"
    flower-superlink --insecure | tee "$LOG_DIR/superlink.log" &
    TLS_FLAG="--insecure"
fi

sleep 2

# ==========================
#  Avvio dei supernode (client FL)
#  con limiti CPU per processo
# ==========================

# Limite thread CPU per PyTorch/BLAS
CPU_THREADS=2 # <-- cambia qui il numero di core per client

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
        --node-config "dataset-path=\"$DATASET\"" \
        | tee "$LOG_FILE" &

    echo "[✓] Avviato client $i su porta $PORT con dataset $DATASET (log: $LOG_FILE)"
done

wait
