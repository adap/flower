# Docker client image 🧪

🧪 = this page covers experimental features that might change in future versions of Flower

This example shows how to dockerize Flower client implememntaion. A detailed explanation of this
example can be found [here](https://flower.dev/docs/framework/how-to-run-flower-using-docker.html#flower-client).

## Build image and install dependencies

```bash
docker build -t flwr-client:0.1.0 -f Dockerfile client-code
pip install -r requirements.txt
docker network create --driver bridge flwr-net
```

## Start the long-running Flower server

```bash
docker run --name flwr-server \
    --rm -p 9091:9091 -p 9092:9092 \
    --network flwr-net \
    flwr/server:1.6.0-py3.11-ubuntu22.04 --insecure
```

## Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
docker run --rm \
    --network flwr-net \
    flwr-client:0.1.0 --insecure --server flwr-server:9092
```

In yet another new terminal window, start the second long-running Flower client:

```bash
docker run --rm \
    --network flwr-net \
    flwr-client:0.1.0 --insecure --server flwr-server:9092
```

## Start the Driver script

```bash
python driver.py
```

## Remove the network

```bash
docker network rm flwr-net
```
