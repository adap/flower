"""fedrs: A Flower Baseline."""

import torch

from fedrs.dataset import load_data
from fedrs.model import get_parameters, initialize_model, set_parameters, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


def _calc_logit_correction(num_classes, labels, device, alpha) -> torch.Tensor:
    class_count = torch.ones(num_classes) * alpha
    labels = labels.unique(sorted=True, return_counts=False, return_inverse=False)
    for c in labels:
        class_count[c] = 1.0
    class_count = class_count.unsqueeze(dim=0).to(device)
    return class_count


class RestrictedSoftmaxCELoss(torch.nn.CrossEntropyLoss):
    """Cross Entropy loss that uses restricted softmax."""

    def __init__(self, num_classes, labels, device, alpha):
        super().__init__()
        self.logits_correction = _calc_logit_correction(
            num_classes, labels, device, alpha
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply logit correction and return cross entropy loss."""
        corrected_logits = logits * self.logits_correction
        return super().forward(corrected_logits, target)


class FlowerClient(NumPyClient):
    """Standard Flower client."""

    def __init__(
        self,
        net,
        trainloader,
        labels,
        local_epochs,
        learning_rate,
        device,
        alpha,
        momentum,
        weight_decay,
    ):
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.criterion = RestrictedSoftmaxCELoss(
            net.classifier.out_features, # num classes
            labels,
            device,
            alpha,
        )

    def get_parameters(self, config):
        """Return the parameters of the current net."""
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        set_parameters(self.net, parameters)
        proximal_mu = config.get('proximal_mu',0.0)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.learning_rate,
            self.criterion,
            self.momentum,
            self.weight_decay,
            proximal_mu,
        )
        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    num_classes = int(context.run_config["num-classes"])
    model_name = str(context.run_config["model-name"])
    net = initialize_model(model_name, num_classes)

    trainloader, labels = load_data(context)

    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = float(context.run_config["alpha"])
    weight_decay = float(context.run_config["weight-decay"])
    momentum = float(context.run_config["momentum"])

    # Return Client instance
    return FlowerClient(
        net,
        trainloader,
        labels,
        local_epochs,
        learning_rate,
        device,
        alpha,
        momentum,
        weight_decay,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
