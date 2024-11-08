"""Generate client for fedht baseline."""

from typing import cast

import torch
import warnings

from flwr.client import Client, NumPyClient
from flwr.common import Context
from omegaconf import DictConfig

from fedmss.utils import set_model_params, get_model_parameters
from sklearn.metrics import log_loss


# UCI-HD client
class UCIHDClient(NumPyClient):
    """Define UCIHDClient class."""

    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        num_features,
        num_classes,
        cfg: DictConfig,
        device
    ) -> None:
        """UCIHD client for UCI-HD experimentation."""
        self.X_train, self.y_train = trainloader
        self.X_test, self.y_test = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = num_features
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

    def fit(self, parameters, config):
        set_model_params(self.model, parameters, self.cfg)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        return get_model_parameters(self.model, self.cfg), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters, self.cfg)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


# client fn for input into simulation
def generate_client_fn_ucihd(
    dataset, num_features, num_classes, model, cfg: DictConfig, device: torch.device
):
    """Generate client function for simulated FL."""

    # def client_fn(cid: int):
    def client_fn(context: Context) -> Client:
        """Define client function for centralized metrics."""
        # Get node_config value to fetch partition_id
        partition_id = cast(int, context.node_config["partition-id"])

        # Load the partition data
        X_train, y_train, X_val, y_val = dataset
        train_dataset = X_train[int(partition_id)], y_train[int(partition_id)]
        test_dataset = X_val[int(partition_id)], y_val[int(partition_id)]
        num_obs = train_dataset[1].shape[0]

        return UCIHDClient(
            train_dataset, test_dataset, model, num_obs, num_features, num_classes, cfg, device
        ).to_client()

    return client_fn
