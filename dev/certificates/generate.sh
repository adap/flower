#!/bin/bash
# This script will generate all certificates if ca.crt does not exist

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

CA_PASSWORD=notsafe

CERT_DIR=.cache/certificates

# Allow passing a custom certificate config file
CERT_CONFIG_FILE="${1:-./dev/certificates/certificate.conf}"

# Generate directories if not exists
mkdir -p .cache/certificates

# if [ -f ".cache/certificates/ca.crt" ]; then
#     echo "Skipping certificate generation as they already exist."
#     exit 0
# fi

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
    -config "$CERT_CONFIG_FILE"

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
    -extfile "$CERT_CONFIG_FILE" \
    -extensions req_ext
