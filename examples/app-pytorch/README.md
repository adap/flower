---
title: Example Flower App using PyTorch
labels: [basic, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower App (PyTorch) 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to start a long-running Flower server (SuperLink) and then run a Flower App (consisting of a `ClientApp` and a `ServerApp`).

## Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
├── client.py           # <-- contains `ClientApp`
├── server.py           # <-- contains `ServerApp`
├── server_workflow.py  # <-- contains `ServerApp` with workflow
├── server_custom.py    # <-- contains `ServerApp` with custom main function
├── task.py             # <-- task-specific code (model, data)
└── requirements.txt    # <-- dependencies
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run a simulation

```bash
flower-simulation --server-app server:app --client-app client:app --num-supernodes 2
```

## Run a deployment

### Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure
```

### Start the long-running Flower client (SuperNode)

In a new terminal window, start the first long-running Flower client:

```bash
flower-client-app client:app --insecure
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client-app client:app --insecure
```

### Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```

Or, to try the workflow example, run:

```bash
flower-server-app server_workflow:app --insecure
```

Or, to try the custom server function example, run:

```bash
flower-server-app server_custom:app --insecure
```
