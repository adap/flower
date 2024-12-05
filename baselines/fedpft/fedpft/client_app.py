"""fedpft: A Flower Baseline."""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from fedpft.dataset import load_data
from fedpft.model import clip_vit, extract_features, resnet50, test, train, transform
from fedpft.utils import gmmparam_to_ndarrays, learn_gmm
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar


class FedPFTClient(NumPyClient):
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


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset = str(context.run_config["dataset"])
    batch_size = int(context.run_config["batch-size"])
    dirichlet_alpha = float(context.run_config["dirichlet-alpha"])
    partition_by = str(context.run_config["partition-by"])
    image_column_name = str(context.run_config["image-column-name"])
    image_input_size = int(context.run_config["image-input-size"])
    seed = int(context.run_config["seed"])

    if dataset == "cifar100":
        trans = transform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        trans = transform(
            [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        )

    client = (
        FedPFTClient if context.run_config["strategy"] == "fedpft" else FedAvgClient
    )
    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        dataset,
        batch_size,
        dirichlet_alpha,
        partition_by,
        image_column_name,
        trans,
        image_input_size,
        seed,
    )
    feature_extractor = (
        clip_vit if context.run_config["feature-extractor"] == "clip_vit" else resnet50
    )
    num_classes = int(context.run_config["num-classes"])
    device = torch.device(str(context.run_config["device"]))

    return client(
        trainloader=trainloader,
        testloader=valloader,
        feature_extractor=feature_extractor(
            str(context.run_config["feature-extractor-name"])
        ),
        num_classes=num_classes,
        device=device,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
