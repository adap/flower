"""fedlc: A Flower Baseline."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedlc.dataset import load_data
from fedlc.model import get_weights, set_weights, test, train, initialize_model


class FedLcClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, learning_rate, device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.net.to(self.device)
    
    def get_parameters(self, config):
        """Return the parameters of the current net."""
        return get_weights(self.net)

    def fit(self, parameters, config):
        """Traim model using this client's data."""
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.learning_rate
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = int(context.run_config["num-classes"])
    num_channels = int(context.run_config["num-channels"])
    model_name = str(context.run_config["model-name"])
    net = initialize_model(model_name, num_channels, num_classes)

    dirichlet_alpha = float(context.run_config["dirichlet-alpha"])
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset = str(context.run_config["dataset"])
    batch_size = int(context.run_config["batch-size"])

    trainloader, valloader = load_data(
        dataset,
        partition_id,
        num_partitions,
        batch_size,
        dirichlet_alpha
    )

    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    # Return Client instance
    return FedLcClient(net, trainloader, valloader, local_epochs, learning_rate, device).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
