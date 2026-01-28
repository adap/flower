#!/usr/bin/env bash

set -e

N=${1:-2}   # number of PostgreSQL databases (default = 2)
BASE_PORT=5433

{
  echo "services:"
  
  for i in $(seq 1 "$N"); do
    PORT=$((BASE_PORT + i - 1))
    # Set a seed for each of the database for producing different random data
    SEED=$(echo "scale=2; $i / 100" | bc)
    cat <<EOF
  postgres_$i:
    image: postgres:18
    container_name: postgres_$i
    environment:
      POSTGRES_USER: flwrlabs
      POSTGRES_PASSWORD: flwrlabs
      POSTGRES_DB: flwrlabs
      DB_SEED: $SEED
    ports:
      - "$PORT:5432"
    volumes:
      - ./db_init.sh:/docker-entrypoint-initdb.d/init.sh:ro

EOF
  done
} | docker compose -f - up -d
