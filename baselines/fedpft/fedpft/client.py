"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from fedpft.models import extract_features, test, train
from fedpft.utils import gmmparam_to_ndarrays, learn_gmm


class FedPFTClient(fl.client.NumPyClient):
    """Flower FedPFTClient."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        feature_extractor: torch.nn.Module,
        num_classes: int,
        device: torch.device,
    ) -> None:
        """FedPFT client strategy.

        Implementation based on https://arxiv.org/abs/2402.01862

        Parameters
        ----------
        trainloader : DataLoader
            Dataset used for learning GMMs
        testloader : DataLoader
            Dataset used for evaluating `classifier_head` sent from the server
        feature_extractor : torch.nn.Module
            Model used to extract features of each client
        num_classes : int
            Number of total classes in the dataset
        device : torch.device
            Device used to extract features and evaluate `classifier_head`
        """
        self.trainloader = trainloader
        self.testloader = testloader
        self.feature_extractor = feature_extractor
        self.classifier_head = nn.Linear(
            feature_extractor.hidden_dimension, num_classes
        )
        self.device = device

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the `classifier_head`."""
        return [
            val.cpu().numpy() for _, val in self.classifier_head.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set the parameters of the `classifier_head`."""
        params_dict = zip(self.classifier_head.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.classifier_head.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Fit a GMM on features and return GMM parameters."""
        # Extracting features
        features, labels = extract_features(
            dataloader=self.trainloader,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )

        # Learning GMM
        gmm_list = learn_gmm(
            features=features,
            labels=labels,
            n_mixtures=int(config["n_mixtures"]),
            cov_type=str(config["cov_type"]),
            seed=int(config["seed"]),
            tol=float(config["tol"]),
            max_iter=int(config["max_iter"]),
        )

        # Reshaping GMM parameters into an NDArray
        return [array for gmm in gmm_list for array in gmmparam_to_ndarrays(gmm)], 0, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate `classifier_head` on the test data."""
        self.set_parameters(parameters)
        loss, acc = test(
            classifier_head=self.classifier_head,
            dataloader=self.testloader,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )
        return loss, len(self.testloader.dataset), {"accuracy": acc}


class FedAvgClient(FedPFTClient):
    """Flower FedAvgClient."""

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the classifier head."""
        self.set_parameters(parameters)

        # train classifier head
        opt = torch.optim.AdamW(
            params=self.classifier_head.parameters(), lr=float(config["lr"])
        )
        train(
            classifier_head=self.classifier_head,
            dataloader=self.trainloader,
            feature_extractor=self.feature_extractor,
            device=self.device,
            num_epochs=int(config["num_epochs"]),
            opt=opt,
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}


# pylint: disable=too-many-arguments
def generate_client_fn(
    client_cfg: DictConfig,
    trainloaders: List[DataLoader],
    testloaders: List[DataLoader],
    feature_extractor: torch.nn.Module,
    num_classes: int,
    device: torch.device,
) -> Callable[[str], fl.client.NumPyClient]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    client_cfg : DictConfig
        Type of client
    trainloaders : List[DataLoader]
        List of train dataloaders for clients
    testloaders : List[DataLoader]
        List of test dataloaders for clients
    feature_extractor : torch.nn.Module
        Pre-trained model as the backbone
    num_classes : int
        Number of classes in the dataset
    device : torch.device
        Device to load the `feature_extractor`
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        """Create a FedPFT client."""
        return instantiate(
            client_cfg,
            trainloader=trainloaders[int(cid)],
            testloader=testloaders[int(cid)],
            feature_extractor=feature_extractor,
            num_classes=num_classes,
            device=device,
        )

    return client_fn
