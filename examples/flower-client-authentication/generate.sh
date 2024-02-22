#!/bin/bash
# This script will generate all server and client credentials

set -e
# Change directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p keys

rm -f keys/*

ssh-keygen -t ecdsa -b 384 -N "" -f "keys/server_credentials"

generate_client_credentials() {
    local num_clients=${1:-2} 
    for ((i=1; i<=num_clients; i++))
    do
        ssh-keygen -t ecdsa -b 384 -N "" -f "keys/client_credentials_$i" -C ""
    done
}

generate_client_credentials $1

printf "%s" "$(cat "keys/client_credentials_1.pub" | sed 's/.$//')" > keys/client_public_keys.csv
for ((i=2; i<=${1:-2}; i++))
do
    printf ",%s" "$(cat "keys/client_credentials_$i.pub" | sed 's/.$//')" >> keys/client_public_keys.csv
done
printf "\n" >> keys/client_public_keys.csv
