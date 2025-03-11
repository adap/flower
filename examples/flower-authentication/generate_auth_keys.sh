#!/bin/bash
# This script will generate keys to setup node authentication

set -e
# Change directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

KEY_DIR=keys

mkdir -p $KEY_DIR

rm -f $KEY_DIR/*

generate_client_credentials() {
    local num_clients=${1:-2} 
    for ((i=1; i<=num_clients; i++))
    do
        ssh-keygen -t ecdsa -b 384 -N "" -f "${KEY_DIR}/client_credentials_$i" -C ""
    done
}

generate_client_credentials "$1"

printf "%s" "$(cat "${KEY_DIR}/client_credentials_1.pub" | sed 's/.$//')" > $KEY_DIR/client_public_keys.csv
for ((i=2; i<=${1:-2}; i++))
do
    printf ",%s" "$(sed 's/.$//' < "${KEY_DIR}/client_credentials_$i.pub")" >> $KEY_DIR/client_public_keys.csv
done
printf "\n" >> $KEY_DIR/client_public_keys.csv
