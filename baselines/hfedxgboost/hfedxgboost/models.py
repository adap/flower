"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from collections import OrderedDict
from typing import Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import NDArray
from hydra.utils import instantiate
from omegaconf import DictConfig
from xgboost import XGBClassifier, XGBRegressor


def fit_xgboost(
    config: DictConfig,
    task_type: str,
    x_train: NDArray,
    y_train: NDArray,
    n_estimators: int,
) -> Union[XGBClassifier, XGBRegressor]:
    """Fit XGBoost model to training data.

    Parameters
    ----------
        config (DictConfig): Hydra configuration.
        task_type (str): Type of task, "REG" for regression and "BINARY"
        for binary classification.
        x_train (NDArray): Input features for training.
        y_train (NDArray): Labels for training.
        n_estimators (int): Number of trees to build.

    Returns
    -------
        Union[XGBClassifier, XGBRegressor]: Fitted XGBoost model.
    """
    if config.dataset.dataset_name == "all":
        if task_type.upper() == "REG":
            tree = instantiate(config.XGBoost.regressor, n_estimators=n_estimators)
        elif task_type.upper() == "BINARY":
            tree = instantiate(config.XGBoost.classifier, n_estimators=n_estimators)
    else:
        tree = instantiate(config.XGBoost)
    tree.fit(x_train, y_train)
    return tree


class CNN(nn.Module):
    """CNN model."""

    def __init__(self, cfg: DictConfig, n_channel: int = 64) -> None:
        super().__init__()
        n_out = 1
        self.task_type = cfg.dataset.task.task_type
        n_estimators_client = cfg.n_estimators_client
        client_num = cfg.client_num

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=n_channel,
            kernel_size=n_estimators_client,
            stride=n_estimators_client,
            padding=0,
        )

        self.layer_direct = nn.Linear(n_channel * client_num, n_out)

        self.relu = nn.ReLU()

        if self.task_type == "BINARY":
            self.final_layer = nn.Sigmoid()
        elif self.task_type == "REG":
            self.final_layer = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass.

        Parameters
        ----------
            input_features (torch.Tensor): Input features to the network.

        Returns
        -------
            output (torch.Tensor): Output of the network after the forward pass.
        """
        output = self.conv1d(input_features)
        output = output.flatten(start_dim=1)
        output = self.relu(output)
        output = self.layer_direct(output)
        output = self.final_layer(output)
        return output

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights.

        Parameters
        ----------
            a list of NumPy arrays.
        """
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set the CNN model weights.

        Parameters
        ----------
            weights:a list of NumPy arrays
        """
        layer_dict = {}
        for key, value in zip(self.state_dict().keys(), weights):
            if value.ndim != 0:
                layer_dict[key] = torch.Tensor(np.array(value, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)
