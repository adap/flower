# $project_name

## Install dependencies

```bash
pip install .
```

## Run (Simulation Engine)

In the `$import_name` directory, use `flwr run` to run a local simulation:

```bash
flwr run
```

## Run (Deployment Engine)

### Start the SuperLink

```bash
flower-superlink --insecure
```

### Start the long-running Flower client

In a new terminal window, start the first long-running Flower client:

```bash
flower-client-app client:app --insecure
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client-app client:app --insecure
```

### Start the ServerApp

```bash
flower-server-app server:app --insecure
```
