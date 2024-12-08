"""fedlc: A Flower Baseline."""

import torch

from fedlc.dataset import load_data
from fedlc.model import get_parameters, CNNModel, set_parameters, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class LogitCorrectedLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        num_classes,
        labels,
        tau,
        device,
    ):
        super().__init__()
        class_count = torch.zeros(num_classes).long()
        labels, counts = labels.unique(
            sorted=True, return_counts=True, return_inverse=False
        )
        class_count[labels] = counts
        class_count = class_count.to(device)
        self.correction = tau * class_count.pow(-0.25)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        corrected_logits = logits - self.correction
        return super().forward(corrected_logits, target)

class FlowerClient(NumPyClient):
    def __init__(
        self,
        net,
        trainloader,
        labels,
        local_epochs,
        tau,
        learning_rate,
        device
    ):
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        use_lc = tau > 0.0
        if use_lc:
            self.criterion = LogitCorrectedLoss(
                net.fc.out_features, # num_classes
                labels,
                tau,
                device,
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """Return the parameters of the current net."""
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.learning_rate,
            self.criterion,
        )
        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    trainloader, labels, num_classes = load_data(context)
    net = CNNModel(num_classes)

    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = float(context.run_config["tau"])

    return FlowerClient(
        net,
        trainloader,
        labels,
        local_epochs,
        tau,
        learning_rate,
        device
    ).to_client()

# Flower ClientApp
app = ClientApp(client_fn)
