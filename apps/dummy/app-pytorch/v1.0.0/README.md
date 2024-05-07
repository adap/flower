## Run a deployment (In the root directory of `flower`)

### Start Flower SuperLink

```bash
flower-superlink --insecure
```

### Start Flower SuperNode

In a new terminal window, start the first Flower SuperNode:

```bash
flower-supernode --insecure --flwr-dir ./
```

In yet another new terminal window, start the second Flower SuperNode:

```bash
flower-supernode --insecure --flwr-dir ./
```

### Run the Flower App

With both the SuperLink and two clients (SuperNode) up and running, we can now run the actual Flower ServerApp:

```bash
flower-server-app server:app --insecure --dir ./apps/dummy/app-pytorch/v1.0.0 --fab-id "dummy/app-pytorch" --fab-version "v1.0.0"
```

