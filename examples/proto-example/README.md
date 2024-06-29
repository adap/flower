# proto-example

> \[!TIP\]
> An example created from `flwr new`'s `PyTorch` template with updated `client_fn` signature.

## Install dependencies

```bash
pip install .
```

## Run (Simulation Engine)

In the `proto-example` directory, use `flwr run` to run a local simulation:

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
flower-supernode proto_example.client:app --insecure --partition-id=0
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode proto_example.client:app --insecure --partition-id=1
```

### Start the Run

```bash
flwr run --use-superexec
```
