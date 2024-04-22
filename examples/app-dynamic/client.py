from typing import Dict

from flwr.client import ClientApp, NumPyClient
from flwr.common import Scalar, Config

from task import (
    DEVICE,
    get_weights,
    set_weights,
    train,
    test,
)
from models import Net, MNISTNet
from datasets import load_cifar_data, load_mnist_data


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __int__(self):
        self.support_dict = {
            "mnist": True,
            "cifar": True,
        }

    def fit(self, parameters, config):
        if config["dataset"] not in self.support_dict.keys():
            return [], 0, {}
        net = self.get_net(config)
        trainloader, testloader = self.get_data(config)
        set_weights(net, parameters)

        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_weights(net), len(trainloader.dataset), results

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Info that can be fetched by the server."""
        return self.support_dict

    def get_net(self, config):
        net_name = config["net"]
        if net_name == "cifar":
            net = Net().to(DEVICE)
        elif net_name == "mnist":
            net = MNISTNet().to(DEVICE)
        else:
            raise ValueError(
                f"This client supports CIFAR and MNIST nets but requested "
                f"{net_name}")
        return net

    def get_data(self, config):
        dataset_name = config["dataset"]
        if dataset_name == "cifar":
            trainloader, testloader = load_cifar_data()
        elif dataset_name == "mnist":
            trainloader, testloader = load_mnist_data()
        else:
            raise ValueError(
                f"This client supports CIFAR and MNIST dataset but requested "
                f"{dataset_name}")
        return trainloader, testloader

    def evaluate(self, parameters, config):
        net = self.get_net(config)
        trainloader, testloader = self.get_data(config)
        set_weights(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
