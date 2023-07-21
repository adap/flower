# Secure Aggregation with Driver API

This example contains highly experimental code. Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced_pytorch)) to learn how to use Flower with PyTorch.

## Setup

```bash
./dev/venv-reset.sh
```

## Run with Driver API

Terminal 1: start Flower server

```bash
flower-server --grpc-rere
```

Terminal 2+3+4: start three clients

```bash
python client.py
```

Terminal 4: start Driver script

Using:

```bash
python driver.py
```
