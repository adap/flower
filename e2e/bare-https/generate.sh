#!/bin/bash
# This script will generate all credentials

set -e
# Change directory to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

CA_PASSWORD=notsafe

CERT_DIR=credentials

# Generate directories if not exists
mkdir -p $CERT_DIR

# Clearing any existing files in the credentials directory
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
    -config credential.conf

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
    -extfile credential.conf \
    -extensions req_ext
