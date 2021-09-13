#!/bin/bash
# This script will re/generate all certificates

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CA_PASSWORD=notsafe

if [ -f "root.key" ]; then
    echo "Skipping certificate generation as they already exist."
    exit 0
fi

rm -f localhost.crt localhost.csr localhost.key root.key root.pem root.srl


# Generate the root certificate authority key with the set password
openssl genrsa \
    -des3 \
    -passout pass:$CA_PASSWORD \
    -out root.key \
    2048

# Generate a root-certificate based on the root-key.
openssl req \
    -x509 \
    -new \
    -nodes \
    -key root.key \
    -passin pass:$CA_PASSWORD \
    -config root.conf \
    -sha256 \
    -out \
    root.pem

# Generate a new private key
openssl genrsa \
    -out localhost.key \
    2048

# Generate a Certificate Signing Request (CSR) based on that private key
openssl req \
    -new \
    -key localhost.key \
    -out localhost.csr \
    -config root.conf

# Create the certificate for the server using the localhost.conf config.
openssl x509 \
    -req \
    -in localhost.csr \
    -CA root.pem \
    -CAkey root.key \
    -CAcreateserial \
    -out localhost.crt \
    -days 1024 \
    -sha256 \
    -extfile localhost.conf \
    -passin pass:$CA_PASSWORD
