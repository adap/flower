---
title: Flower Example with Authentication
tags: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Authentication with PyTorch 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to start a long-running Flower server (SuperLink) and a long-running Flower client (SuperNode) with authentication enabled.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp && mv _tmp/examples/flower-authentication . && rm -rf _tmp && cd flower-authentication
```

This will create a new directory called `flower-authentication` with the following project structure:

```bash
$ tree .
.
├── certificate.conf  # <-- configuration for OpenSSL
├── generate.sh       # <-- generate certificates and keys
├── pyproject.toml    # <-- project dependencies
├── client.py         # <-- contains `ClientApp`
├── server.py         # <-- contains `ServerApp`
└── task.py           # <-- task-specific code (model, data)
```

## Install dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Generate public and private keys

```bash
./generate.sh
```

`generate.sh` is a script that (by default) generates certificates for creating a secure TLS connection
and three private and public key pairs for one server and two clients.
You can generate more keys by specifying the number of client credentials that you wish to generate.
The script also generates a CSV file that includes each of the generated (client) public keys.

⚠️ Note that this script should only be used for development purposes and not for creating production key pairs.

```bash
./generate.sh {your_number_of_clients}
```

## Start the long-running Flower server (SuperLink)

To start a long-running Flower server (SuperLink) and enable authentication is very easy; all you need to do is type
`--auth-list-public-keys` containing file path to the known `client_public_keys.csv`, `--auth-superlink-private-key`
containing file path to the SuperLink's private key `server_credentials`, and `--auth-superlink-public-key` containing file path to the SuperLink's public key `server_credentials.pub`. Notice that you can only enable authentication with a secure TLS connection.

```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --auth-list-public-keys keys/client_public_keys.csv \
    --auth-superlink-private-key keys/server_credentials \
    --auth-superlink-public-key keys/server_credentials.pub
```

## Start the long-running Flower client (SuperNode)

In a new terminal window, start the first long-running Flower client (SuperNode):

```bash
flower-client-app client:app \
    --root-certificates certificates/ca.crt \
    --server 127.0.0.1:9092 \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client-app client:app \
    --root-certificates certificates/ca.crt \
    --server 127.0.0.1:9092 \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub
```

If you generated more than 2 client credentials, you can add more clients by opening new terminal windows and running the command
above. Don't forget to specify the correct client private and public keys for each client instance you created.

## Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower ServerApp:

```bash
flower-server-app server:app --root-certificates certificates/ca.crt --dir ./ --server 127.0.0.1:9091
```
