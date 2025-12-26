---
tags: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Federations with Authentication 🧪

The following steps describe how to start a long-running Flower server (SuperLink) and a long-running Flower clients (SuperNode) with authentication enabled. The task is to train a simple CNN for image classification using PyTorch.

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to learn more about Flower's Deployment Engine, how setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) works. If you are already familiar with how the Deployment Engine works, you may want to learn how to run this same example using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/flower-authentication . \
        && rm -rf _tmp && cd flower-authentication
```

This will create a new directory called `flower-authentication` with the following project structure:

```shell
flower-authentication
├── authexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── certificate.conf    # Configuration for OpenSSL
├── generate.sh         # Generate certificates and keys
├── prepare_dataset.py  # Generate datasets for each SuperNode to use
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `authexample` package.

```bash
pip install -e .
```

## Generate TLS certificates

The `generate_cert.sh` script generates certificates for creating a secure TLS connection between the SuperLink and SuperNodes, as well as between the flwr CLI (user) and the SuperLink.

> [!NOTE]
> Note that this script should only be used for development purposes and not for creating production key pairs.

```bash
./generate_cert.sh
```

## Generate public and private keys for SuperNode authentication

The `generate_auth_keys.sh` script generates two private–public key pairs for two SuperNodes by default. If you have more SuperNodes, you can specify the number of key pairs to generate.

> [!NOTE]
> Note that this script should only be used for development purposes and not for creating production key pairs.

```bash
# Generate two key pairs by default
./generate_auth_keys.sh

# Or pass the desired the number of key pairs
./generate_auth_keys.sh {your_number_of_clients}
```

## Start the long-running Flower server (SuperLink)

Starting long-running Flower server component (SuperLink) and enable authentication is very easy; all you need to do is to pass the `--enable-supernode-auth` flag. In this example we also enable secure TLS communications between `SuperLink`, the `SuperNodes` and the Flower CLI.

Let's first launch the `SuperLink`:

```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --enable-supernode-auth
```

At this point your server-side is idling. Next, let's connect two `SuperNode`s, and then we'll start a run.

## Start the long-running Flower client (SuperNode)

> [!NOTE]
> Typically each `SuperNode` runs in a different entity/organization which has access to a dataset. In this example we are going to artificially create N dataset splits and saved them into a new directory called `datasets/`. Then, each `SuperNode` will be pointed to the dataset it should load via the `--node-config` argument. We provide a script that does the download, partition and saving of CIFAR-10.

```bash
python prepare_dataset.py
```

### Pre-registering SuperNodes

Before connecting the `SuperNodes` we need to register them with the `SuperLink`. This means we'll tell the `SuperLink` about the identities of the `SuperNodes` that will be connected. We do this by sending to it the public keys of the `SuperNodes` that we want the `SuperLink` to authorize.

Let's register the first `SuperNode`. The command below will send the public key to the `SuperLink` defined in the `my-federation` federation in the `pyproject.toml`.

```shell
flwr supernode register keys/client_credentials_1.pub . my-federation
# It will print something like:
# Loading project configuration...
# Success
# ✅ SuperNode 16019329408659850374 registered successfully.
```

Then, we register the second `SuperNode` using the other public key:

```shell
flwr supernode register keys/client_credentials_2.pub . my-federation
# It will print something like:
# Loading project configuration...
# Success
# ✅ SuperNode 8392976743692794070 registered successfully.
```

You could also use the Flower ClI to view the status of the `SuperNodes`.

```shell
flwr supernode list . my-federation
📄 Listing all nodes...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃       Node ID        ┃   Owner    ┃ Status  ┃ Elapsed  ┃   Status Changed @   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 16019329408659850374 │<name:none> │ created │          │ N/A                  │
├──────────────────────┼────────────┼─────────┼──────────┼──────────────────────┤
│ 8392976743692794070  │<name:none> │ created │          │ N/A                  │
└──────────────────────┴────────────┴─────────┴──────────┴──────────────────────┘
```

Once the `SuperNodes` are connected, you'll see the status changes. Let's connect them !

### Connecting SuperNodes

In a new terminal window, start the first long-running Flower client (SuperNode):

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_1 \
    --superlink "127.0.0.1:9092" \
    --node-config 'dataset-path="datasets/cifar10_part_1"' \
    --clientappio-api-address="0.0.0.0:9094"
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_2 \
    --superlink "127.0.0.1:9092" \
    --node-config 'dataset-path="datasets/cifar10_part_2"' \
    --clientappio-api-address="0.0.0.0:9095"
```

Now that you have connected the `SuperNodes`, you should see them with status `online`:

```shell
flwr supernode list . my-federation
📄 Listing all nodes...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃       Node ID        ┃   Owner    ┃ Status  ┃ Elapsed  ┃   Status Changed @   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 16019329408659850374 │<name:none> │ online  │ 00:00:30 │ 2025-10-13 13:40:47Z │
├──────────────────────┼────────────┼─────────┼──────────┼──────────────────────┤
│ 8392976743692794070  │<name:none> │ online  │ 00:00:22 │ 2025-10-13 13:52:21Z │
└──────────────────────┴────────────┴─────────┴──────────┴──────────────────────┘
```

If you generated more than 2 client credentials, you can add more clients by opening new terminal windows and running the command
above. Don't forget to specify the correct client private key for each client (SuperNode) you created.

> [!TIP]
> Note the `--node-config` passed when spawning the `SuperNode` is accessible to the `ClientApp` via the `context` argument, i.e., `context.node_config`. In this example, the `ClientApp` uses it to load the dataset and then proceed with the training of the model.
>
> ```python
> @app.train()
> def train(msg: Message, context: Context):
>     # Read the node_config to know where dataset is located
>     dataset_path = context.node_config["dataset-path"]
>     # then load the dataset
> ```

## Run the Flower App

With both the long-running server (SuperLink) and two SuperNodes up and running, we can now start the run. Note that the command below points to a federation named `my-federation`. Its entry point is defined in the `pyproject.toml`. You can optionally use the `--stream` flag to stream logs from your `ServerApp` running on SuperLink.

```bash
flwr run . my-federation --stream
```
