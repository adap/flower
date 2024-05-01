# Flower Client Authentication with PyTorch 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch), [advanced](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to start a long-running Flower server (SuperLink) and then run a Flower App (consisting of a `ClientApp` and a `ServerApp`) with client authentication enabled.

## Preconditions

Let's assume the following project structure:

```bash
$ tree .
.
├── pyproject.toml    # <-- project dependencies
├── client.py         # <-- contains `ClientApp`
├── server.py         # <-- contains `ServerApp`
└── task.py           # <-- task-specific code (model, data)
```

## Install dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

## Generate public and private keys

```bash
bash ./generate.sh
```

`generate.sh` is a script that generates per default three private and public keys, for server and two clients.
You can generate more keys by specifying number of client credentials that you wish.
Also, the script generates a csv file that includes each of the generated client credentials public keys.

```bash
bash ./generate.sh {your_number_here}
```

## Start the long-running Flower server (SuperLink)

```bash
flower-superlink --insecure --require-client-authentication ./keys/client_public_keys.csv ./keys/server_credentials.pub ./keys/server_credentials
```

To start a long-running Flower server and enable client authentication is very easy, all you need to do is to type
`--require-client-authentication` followed by the path to the known `client_public_keys.csv`, server's public key
`server_credentials.pub`, and server's private key `server_credentials`.

## Start the long-running Flower client (SuperNode)

In a new terminal window, start the first long-running Flower client:

```bash
flower-client-app client:app --insecure --authentication-keys ./keys/client_credentials_1.pub ./keys/client_credentials_1
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-client-app client:app --insecure --authentication-keys ./keys/client_credentials_2.pub ./keys/client_credentials_2
```

If you generated more than 2 client credentials, you can add more clients by opening new terminal window and run the command
above, don't forget to specify the correct client public and private keys for each client instance you created.

## Run the Flower App

With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```

Or, to try the custom server function example, run:

```bash
flower-server-app server_custom:app --insecure
```
