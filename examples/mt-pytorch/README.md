# Multi-Tenant Federated Learning with Flower and PyTorch

This example contains experimental code. Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

## Setup

```bash
./dev/venv-reset.sh
```

## Run with Driver API

Terminal 1: start Flower server

```bash
flower-server
```

Terminal 2+3: start two Flower client nodes

```bash
python client.py
```

Terminal 4: start Driver script

Using:

```bash
python start_driver.py
```

Or, alternatively:

```bash
python driver.py
```

## Run in legacy mode

Terminal 1: start Flower server

```bash
python server.py
```

Terminal 2+3: start two clients

```bash
python client.py
```
