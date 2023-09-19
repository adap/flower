# Deploy 🧪

🧪 = this page covers experimental features that might change in future versions of Flower

This how-to guide describes the deployment of a long-running Flower server.

## Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
└── client.py
├── driver.py
├── requirements.txt
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start the long-running Flower server

```bash
flower-server --grpc-rere
```

## Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
flower-client --grpc-rere --app client:app
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client --grpc-rere --app client:app
```

## Start the Driver script

```bash
python driver.py
```
