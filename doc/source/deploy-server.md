# Deploy a Flower Server 🧪

🧪 = this page covers experimental features that might change in future versions of Flower

This how-to guide describes the deployment of a long-running Flower server.

## Start Flower server

### ... using the gRPC transport layer

Start Driver API:

```bash
flower-driver-api --driver-api-address "0.0.0.0:9091" --database flwr.db
```

Start Flower Fleet API:

```bash
flower-fleet-api --rest --rest-fleet-api-address "0.0.0.0:9093" --database flwr.db
```

Or, start them both together:

```bash
flower-server --rest --rest-fleet-api-address "0.0.0.0:9093" --database flwr.db
```

### ... using the (experimental) REST transport layer

Start Driver API:

```bash
flower-driver-api --driver-api-address "0.0.0.0:9091" --database flwr.db
```

Start Flower Fleet API:

```bash
flower-fleet-api --rest --rest-fleet-api-address "0.0.0.0:9093" --database flwr.db
```

Or, start them both together:

```bash
flower-server --rest --rest-fleet-api-address "0.0.0.0:9093" --database flwr.db
```

## Start Client Node(s)

### ... connecting to the gRPC Fleet API

In `examples/mt-pytorch`:

1. Open `client.py` and change `rest=True` to `rest=False`
2. Run `python client.py` in three separate terminal windows to start three clients

### ... connecting to the REST Fleet API

In `examples/mt-pytorch`:

1. Open `client.py` and change `rest=True` to `rest=False`
2. Run `python client.py` in three separate terminal windows to start three clients

## Run Driver Script

To test whether everything is running as expected, start a driver script.

In `examples/mt-pytorch`:

1. Open `driver.py` and verify that the `Driver` class is initialized with the correct Driver API address (host and port)
2. Run `python driver.py` in a separate terminal window to start the driver script
