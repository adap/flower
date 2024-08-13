#!/bin/bash
# This script will generate all certificates if ca.crt does not exist

set -e
# Change directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

CERT_DIR=certificates

# Generate directories if not exists
mkdir -p $CERT_DIR

# Clearing any existing files in the certificates directory
rm -f $CERT_DIR/*

# Generate the root certificate authority key and certificate based on key
openssl genrsa -out $CERT_DIR/ca.key 4096
openssl req \
    -new \
    -x509 \
    -key $CERT_DIR/ca.key \
    -sha256 \
    -subj "/C=DE/ST=HH/O=CA, Inc." \
    -days 365 -out $CERT_DIR/ca.crt

# Generate a new private key for the server
openssl genrsa -out $CERT_DIR/server.key 4096

# Create a signing CSR
openssl req \
    -new \
    -key $CERT_DIR/server.key \
    -out $CERT_DIR/server.csr \
    -config certificate.conf

# Generate a certificate for the server
openssl x509 \
    -req \
    -in $CERT_DIR/server.csr \
    -CA $CERT_DIR/ca.crt \
    -CAkey $CERT_DIR/ca.key \
    -CAcreateserial \
    -out $CERT_DIR/server.pem \
    -days 365 \
    -sha256 \
    -extfile certificate.conf \
    -extensions req_ext

KEY_DIR=keys

mkdir -p $KEY_DIR

rm -f $KEY_DIR/*

ssh-keygen -t ecdsa -b 384 -N "" -f "${KEY_DIR}/server_credentials" -C ""

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
