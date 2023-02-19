# 🧑‍🔬 Deploy a Flower Server

This how-to guide describes the deployment of a Flower server.

## Start Flower server

Start Driver API:

```bash
flower-driver-api --driver-api-address "0.0.0.0:9091" --database-path flwr.db
```

Start Flower Fleet API:

```bash
flower-fleet-api --rest --rest-fleet-api-address "0.0.0.0:9093" --database-path flwr.db
```

Or, start them both together:

```bash
flower-server --rest --rest-fleet-api-address "0.0.0.0:9093" --database-path flwr.db
```

## Start Client Node(s)

Using the `mt-pytorch` code example, run

```bash
python client.py
```

in three different terminal windows.

## Run Driver Script

Using the `mt-pytorch` code example, run

```bash
python driver.py
```

in yet another terminal window.
