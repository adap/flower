# Multi-Tenant Federated Learning with Flower and PyTorch

This example contains highly experimental code. Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced_pytorch)) to learn how to use Flower with PyTorch.

## Setup

```bash
./dev/venv-reset.sh
```

## Exec

Terminal 1: start Driver API server

```bash
flower-server
```

Terminal 2: run driver script

```bash
python driver.py
```
