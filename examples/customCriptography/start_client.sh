#!/bin/bash

NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto  import NUM_CLIENTS; print(NUM_CLIENTS)")
# Recupero variabili da Python
TLS=$(python3 -c "from flwr.common.crypto.config_cripto import TLS; print(TLS)")
NUM_CLIENTS=$(python3 -c "from flwr.common.crypto.config_cripto import NUM_CLIENTS; print(NUM_CLIENTS)")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

mkdir -p $LOG_DIR
# Avvio del superlink
if [ "$TLS" = "True" ]; then
    echo "[*] Avvio superlink in modalità TLS"
    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        >"$LOG_DIR/superlink.log" 2>&1 &
    TLS_FLAG="--root-certificates certificates/ca.crt"
else
    echo "[*] Avvio superlink in modalità INSECURE"
    flower-superlink --insecure >"$LOG_DIR/superlink.log" 2>&1 &
    TLS_FLAG="--insecure"
fi

sleep 2

# Avvio dei supernode
for ((i=1; i<=NUM_CLIENTS; i++)); do
    PORT=$((9093 + i))
    DATASET="datasets/cifar10_part_$i"
    LOG_FILE="$LOG_DIR/client_$i.log"

    flower-supernode \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 0.0.0.0:${PORT} \
      --insecure \
      $TLS_FLAG \
      --node-config "dataset-path=\"$DATASET\"" \
      >"$LOG_FILE" 2>&1 &

    echo "[✓] Avviato client $i su porta $PORT con dataset $DATASET (log: $LOG_FILE)"
done

wait

