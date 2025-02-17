---
tags: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Federations with Authentication 🧪

> \[!NOTE\]
> 🧪 = This example covers experimental features that might change in future versions of Flower.
> Please consult the regular PyTorch examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to start a long-running Flower server (SuperLink) and a long-running Flower clients (SuperNode) with authentication enabled. The task is to train a simple CNN for image classification using PyTorch.

> \[!TIP\]
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

> \[!NOTE\]
> Note that this script should only be used for development purposes and not for creating production key pairs.

```bash
./generate_cert.sh
```

## Generate public and private keys for SuperNode authentication

The `generate_auth_keys.sh` script generates three private and public key pairs. One pair for the SuperLink and two pairs for the two SuperNodes.

> \[!NOTE\]
> Note that this script should only be used for development purposes and not for creating production key pairs.

```bash
./generate_auth_keys.sh
```

You can generate more keys by specifying the number of client credentials that you wish to generate.
The script also generates a CSV file that includes each of the generated (client) public keys.

```bash
./generate_auth_keys.sh {your_number_of_clients}
```

## Start the long-running Flower server (SuperLink)

Starting long-running Flower server component (SuperLink) and enable authentication is very easy; all you need to do is type
`--auth-list-public-keys` containing file path to the known `client_public_keys.csv`. Notice that you can only enable authentication with a secure TLS connection.

Let's first launch the `SuperLink`:

```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --auth-list-public-keys keys/client_public_keys.csv
```

At this point your server-side is idling. Next, let's connect two `SuperNode`s, and then we'll start a run.

## Start the long-running Flower client (SuperNode)

> \[!NOTE\]
> Typically each `SuperNode` runs in a different entity/organization which has access to a dataset. In this example we are going to artificially create N dataset splits and saved them into a new directory called `datasets/`. Then, each `SuperNode` will be pointed to the dataset it should load via the `--node-config` argument. We provide a script that does the download, partition and saving of CIFAR-10.

```bash
python prepare_dataset.py
```

In a new terminal window, start the first long-running Flower client (SuperNode):

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_1 \
    --auth-supernode-public-key keys/client_credentials_1.pub \
    --node-config 'dataset-path="datasets/cifar10_part_1"' \
    --clientappio-api-address="0.0.0.0:9094"
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/client_credentials_2 \
    --auth-supernode-public-key keys/client_credentials_2.pub \
    --node-config 'dataset-path="datasets/cifar10_part_2"' \
    --clientappio-api-address="0.0.0.0:9095"
```

If you generated more than 2 client credentials, you can add more clients by opening new terminal windows and running the command
above. Don't forget to specify the correct client private and public keys for each client instance you created.

> \[!TIP\]
> Note the `--node-config` passed when spawning the `SuperNode` is accessible to the `ClientApp` via the context. In this example, the `client_fn()` uses it to load the dataset and then proceed with the training of the model.
>
> ```python
> def client_fn(context: Context):
>     # retrieve the passed `--node-config`
>     dataset_path = context.node_config["dataset-path"]
>     # then load the dataset
> ```

## Run the Flower App

With both the long-running server (SuperLink) and two SuperNodes up and running, we can now start the run. Note that the command below points to a federation named `my-federation`. Its entry point is defined in the `pyproject.toml`.

```bash
flwr run . my-federation
```
