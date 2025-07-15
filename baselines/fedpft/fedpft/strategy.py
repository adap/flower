"""FedPFT strategy."""

from typing import Dict, List, Optional, Tuple, Union

import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture as GMM
from torch.utils.data import DataLoader

from fedpft.models import train
from fedpft.utils import chunks, ndarrays_to_gmmparam


class FedPFT(FedAvg):
    """Implementation of FedPFT.

    https://arxiv.org/abs/2402.01862
    Authors:
        Mahdi Beitollahi, Alex Bie, Sobhan Hemati, Leo Maxime Brunswic,
        Xu Li, Xi Chen, Guojun Zhang.
    """

    def __init__(
        self,
        *args,
        num_classes: int,
        feature_dimension: int,
        server_opt: DictConfig,
        server_batch_size: int,
        num_epochs: int,
        device: torch.device,
        **kwargs,
    ) -> None:
        """Create FedPFT strategy.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.
        feature_dimension : int
            Size of feature embeddings
        server_opt : DictConfig
            Configuration of server optimizer for training classifier head.
        server_batch_size : int
            Batch size of synthetic features.
        num_epochs : int
            Number of epochs to train the classifier head.

        Attributes
        ----------
        device : torch.device()
            Device to train the classifier head at the server.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.server_opt = server_opt
        self.server_batch_size = server_batch_size
        self.num_epochs = num_epochs
        self.device = device

    # pylint: disable=too-many-locals
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Learn a classifier head by generating samples from the GMMs."""
        # Do not aggregate if there are failures.
        if not self.accept_failures and failures:
            raise Exception("there are failures and failures are not accepted")

        assert self.on_fit_config_fn is not None
        config = self.on_fit_config_fn(server_round)

        # Sample from the GMMs to create synthetic feature dataset
        synthetic_features_dataset: List[Union[Dict, Tuple]] = []
        for _, fit_res in results:
            # Convert byte parameters into ndarrays and GMMParameters
            ndarray = parameters_to_ndarrays(fit_res.parameters)
            all_gmm_parameters = [
                ndarrays_to_gmmparam(array) for array in chunks(ndarray, 5)
            ]

            # Sample from GMM_label pairs to create synthetic features
            for gmm_parameter in all_gmm_parameters:
                gmm = GMM(
                    n_components=int(config["n_mixtures"]),
                    covariance_type=config["cov_type"],
                    random_state=int(config["seed"]),
                    tol=float(config["tol"]),
                    max_iter=int(config["max_iter"]),
                )
                # Set values of the GMMs
                gmm.means_ = gmm_parameter.means.astype("float32")
                gmm.weights_ = gmm_parameter.weights.astype("float32")
                gmm.covariances_ = gmm_parameter.covariances.astype("float32")

                # Sample features
                syn_features, _ = gmm.sample(gmm_parameter.num_samples)
                syn_features = torch.tensor(syn_features, dtype=torch.float32)
                gmm_labels = torch.tensor(
                    [int(gmm_parameter.label)] * int(gmm_parameter.num_samples)
                )

                # Add to train data
                synthetic_features_dataset += list(zip(syn_features, gmm_labels))

        # Train a classifier head
        synthetic_features_dataset = [
            {"img": img, "label": label} for img, label in synthetic_features_dataset
        ]
        synthetic_loader = DataLoader(
            synthetic_features_dataset,
            batch_size=self.server_batch_size,
            shuffle=True,
        )
        classifier_head = torch.nn.Linear(self.feature_dimension, self.num_classes)
        opt = torch.optim.AdamW(
            params=classifier_head.parameters(), lr=self.server_opt.lr
        )

        train(
            classifier_head=classifier_head,
            dataloader=synthetic_loader,
            device=self.device,
            num_epochs=self.num_epochs,
            opt=opt,
            verbose=True,
        )

        # Send the classifier head to clients
        classifier_ndarray = [
            val.cpu().numpy() for _, val in classifier_head.state_dict().items()
        ]

        return ndarrays_to_parameters(classifier_ndarray), {}
