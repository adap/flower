"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
from collections import OrderedDict
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import Dataset
from flwr.common import NDArray, NDArrays
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig
from hydra.utils import instantiate
import flwr as fl
from flwr.common.typing import Parameters
import numpy as np
import torch
import torch.nn as nn


def fit_XGBoost(
    config: DictConfig, task_type:str, X_train: NDArray,y_train: NDArray, n_estimators: int
) -> Union[XGBClassifier, XGBRegressor]:
    if config.centralized:
        if task_type.upper() == "REG":
            tree = instantiate(config.XGBoost.regressor,n_estimators=n_estimators)
        elif task_type.upper() == "BINARY":
            tree = instantiate(config.XGBoost.classifier,n_estimators=n_estimators)
    else:
        tree = instantiate(config.XGBoost)
    tree.fit(X_train, y_train)
    return tree

class CNN(nn.Module):
    def __init__(self,cfg, n_channel: int = 64) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.task_type = cfg.dataset.task_type
        n_estimators_client=cfg.n_estimators_client
        client_num=cfg.client_num
        self.conv1d = nn.Conv1d(
            1, n_channel, kernel_size=n_estimators_client, stride=n_estimators_client, padding=0
        )
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()

        if self.task_type == "BINARY":
           self.final_layer  = nn.Sigmoid()
        elif self.task_type == "REG":
           self.final_layer  = nn.Identity()
        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1d(x))
        x = x.flatten(start_dim=1)
        x = self.ReLU(x)
        x=self.layer_direct(x)
        x=self.final_layer(x)
        return x


    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)

