#!/bin/bash
# This script will re/generate all certificates

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CA_PASSWORD=notsafe

if [ -f "ca.crt" ]; then
    echo "Skipping certificate generation as they already exist."
    exit 0
fi

rm -f *.crt *.csr *.key *.key *.pem *.srl


# Generate the root certificate authority key and certificate based on key
openssl genrsa -out ca.key 4096
openssl req -new -x509 -key ca.key -sha256 -subj "/C=DE/ST=HH/O=CA, Inc." -days 365 -out ca.crt

# Generate a new private key for the server
openssl genrsa -out server.key 4096

# Create a signing CSR
openssl req -new -key server.key -out server.csr -config certificate.conf

# Generate a certificate for the server
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.pem \
        -days 365 -sha256 -extfile certificate.conf -extensions req_ext
