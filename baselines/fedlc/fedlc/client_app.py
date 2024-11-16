"""fedlc: A Flower Baseline."""

import torch

from fedlc.dataset import load_data
from fedlc.model import get_parameters, initialize_model, set_parameters, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class LogitCorrectedLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, logits_correction: torch.Tensor):
        super().__init__()
        self.logits_correction = logits_correction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Modify the logits before cross entropy loss
        corrected_logits = logits - self.logits_correction
        return super().forward(corrected_logits, target)


def calc_logit_correction(net, labels, device) -> torch.Tensor:
    num_classes = net.fc.out_features
    class_count = torch.zeros(num_classes).long()
    labels, counts = labels.unique(
        sorted=True, return_counts=True, return_inverse=False
    )
    class_count[labels] = counts
    class_count = class_count.to(device)
    return class_count


class FlowerClient(NumPyClient):
    def __init__(
        self, net, trainloader, labels, local_epochs, use_lc, tau, learning_rate, device
    ):
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.use_lc = use_lc
        self.device = device
        if self.use_lc:
            logits_correction = tau * calc_logit_correction(net, labels, device).pow(-0.25)
            self.criterion = LogitCorrectedLoss(logits_correction)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = int(context.run_config["num-classes"])
    num_channels = int(context.run_config["num-channels"])
    model_name = str(context.run_config["model-name"])
    net = initialize_model(model_name, num_channels, num_classes)

    trainloader, labels = load_data(context)

    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    tau = float(context.run_config["tau"])
    use_lc = bool(context.run_config["use-logit-correction"])

    # Return Client instance
    return FlowerClient(
        net, trainloader, labels, local_epochs, use_lc, tau, learning_rate, device
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
