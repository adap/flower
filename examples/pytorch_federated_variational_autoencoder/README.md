# PyTorch: Federated Variational Autoencoder

This example demonstrates how a variational autoencoder (VAE) can be federated using the Flower framework.

## Project Setup

1. Create a virtual environment `python3 -m venv env`.
2. Activate the virtual environment using `source env/bin/activate`.
3. Install all requirements using `pip install -r requirements.txt`.

## Federating the Variational Autoencoder Model

Start the server in a terminal as follows:

```shell
python3 server.py
```

Now that the server is running and waiting for clients, we can start two clients that will participate in the federated learning process. To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py
```

Start client 2 in the second terminal:

```shell
python3 client.py
```