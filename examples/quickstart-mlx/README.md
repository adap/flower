---
title: Simple Flower Example using MLX
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Flower Example using MLX

> \[!TIP\]
> An example created from `flwr new`'s `MLX` template with updated `client_fn` signature.

## Install dependencies

```bash
pip install .
```

## Run (Simulation Engine)

```bash
flwr run
```

## Run (Deployment Engine)

### Start the SuperExec

```bash
flower-superexec flwr.superexec.deployment:executor --insecure
```

### Start the SuperLink

```bash
flower-superlink --insecure
```

### Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
flower-supernode mlxexample.client:app --insecure --partition-id=0
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode mlxexample.client:app --insecure --partition-id=1
```

### Start the Run

```bash
flwr run --use-superexec
```
