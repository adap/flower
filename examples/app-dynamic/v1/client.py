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
from models import CIFARNet, MNISTNet, BiggerCIFARNet
from datasets import load_cifar_data, load_mnist_data


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, dataset_support_dict, model_support_dict):
        self.dataset_support_dict = dataset_support_dict
        self.model_support_dict = model_support_dict

    def fit(self, parameters, config):
        net = self.get_net(config)
        trainloader, testloader = self.get_data(config)
        set_weights(net, parameters)

        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)
        return get_weights(net), len(trainloader.dataset), results

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Info that can be fetched by the server."""
        return {**self.dataset_support_dict, **self.model_support_dict}

    def get_net(self, config):
        net_name = config["net"]
        if net_name == "cifar_net":
            net = CIFARNet().to(DEVICE)
        elif net_name == "bigger_cifar_net":
            net = BiggerCIFARNet().to(DEVICE)
        elif net_name == "mnist_net":
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
    dataset_support_dict = {
        "mnist": True,
        "cifar": True,
    }
    model_support_dict = {
        "mnist_net": True,
        "cifar_net": True,
        "bigger_cifar_net": True
    }
    return FlowerClient(dataset_support_dict, model_support_dict).to_client()
def client_fn_cifar(cid: str):
    """Create and return an instance of Flower `Client`."""
    dataset_support_dict = {
        "mnist": False,
        "cifar": True,
    }
    model_support_dict = {
        "mnist_net": False,
        "cifar_net": True,
        "bigger_cifar_net": True
    }
    return FlowerClient(dataset_support_dict, model_support_dict).to_client()


def client_fn_mnist(cid: str):
    """Create and return an instance of Flower `Client`."""
    dataset_support_dict = {
        "mnist": True,
        "cifar": False,
    }
    model_support_dict = {
        "mnist_net": True,
        "cifar_net": False,
        "bigger_cifar_net": False
    }
    return FlowerClient(dataset_support_dict, model_support_dict).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)

app_cifar = ClientApp(
    client_fn=client_fn_cifar,
)

app_mnist = ClientApp(
    client_fn=client_fn_mnist,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    dataset_support_dict = {
        "mnist": True,
        "cifar": False,
    }
    model_support_dict = {
        "mnist_net": False,
        "cifar_net": True,
        "bigger_cifar_net": True
    }
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(dataset_support_dict, model_support_dict).to_client(),
    )
