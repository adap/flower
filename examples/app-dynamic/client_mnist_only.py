from typing import Dict

from flwr.client import ClientApp

from client import FlowerClient

# SUPPORT_DICT IS DIFFERENT
support_dict = {
    "mnist": True,
    "cifar": False,
}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(support_dict).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(support_dict).to_client(),
    )
