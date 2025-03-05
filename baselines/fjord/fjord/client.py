"""Flower client implementing FjORD."""

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .dataset import load_data
from .models import get_net, test, train
from .od.layers import ODBatchNorm2d, ODConv2d, ODLinear
from .od.samplers import ODSampler
from .utils.logger import Logger
from .utils.utils import save_model

FJORD_CONFIG_TYPE = Dict[
    Union[str, float],
    List[Any],
]


def get_layer_from_state_dict(model: Module, state_dict_key: str) -> Module:
    """Get the layer corresponding to the given state dict key.

    :param model: The model.
    :param state_dict_key: The state dict key.
    :return: The module corresponding to the given state dict key.
    """
    keys = state_dict_key.split(".")
    module = model
    # The last keyc orresponds to the parameter name
    # (e.g., weight or bias)
    for key in keys[:-1]:
        module = getattr(module, key)
    return module


def net_to_state_dict_layers(net: Module) -> List[Module]:
    """Get the state_dict of the model.

    :param net: The model.
    :return: The state_dict of the model.
    """
    layers = []
    for key, _ in net.state_dict().items():
        layer = get_layer_from_state_dict(net, key)
        layers.append(layer)
    return layers


def get_agg_config(
    net: Module, trainloader: DataLoader, p_s: List[float]
) -> FJORD_CONFIG_TYPE:
    """Get the aggregation configuration of the model.

    :param net: The model.
    :param trainloader: The training set.
    :param p_s: The p values used
    :return: The aggregation configuration of the model.
    """
    Logger.get().info("Constructing OD model configuration for aggregation.")
    device = next(net.parameters()).device
    images, _ = next(iter(trainloader))
    images = images.to(device)
    layers = net_to_state_dict_layers(net)
    # init min dims in networks
    config: FJORD_CONFIG_TYPE = {p: [{} for _ in layers] for p in p_s}
    config["layer"] = []
    config["layer_p"] = []
    with torch.no_grad():
        for p in p_s:
            max_sampler = ODSampler(
                p_s=[p],
                max_p=p,
                model=net,
            )
            net(images, sampler=max_sampler)
            for i, layer in enumerate(layers):
                if isinstance(layer, (ODConv2d, ODLinear)):
                    config[p][i]["in_dim"] = layer.last_input_dim
                    config[p][i]["out_dim"] = layer.last_output_dim
                elif isinstance(layer, ODBatchNorm2d):
                    config[p][i]["in_dim"] = None
                    config[p][i]["out_dim"] = layer.p_to_num_features[p]
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    pass
                else:
                    raise ValueError(f"Unsupported layer {layer.__class__.__name__}")
    for layer in layers:
        config["layer"].append(layer.__class__.__name__)
        if hasattr(layer, "p"):
            config["layer_p"].append(layer.p)
        else:
            config["layer_p"].append(None)
    return config


# Define Flower client
class FjORDClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Flower client training on CIFAR-10."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cid: int,
        model_name: str,
        model_path: str,
        data_path: str,
        know_distill: bool,
        max_p: float,
        p_s: List[float],
        train_config: SimpleNamespace,
        fjord_config: FJORD_CONFIG_TYPE,
        log_config: Dict[str, str],
        device: torch.device,
        seed: int,
    ) -> None:
        """Initialise the client.

        :param cid: The client ID.
        :param model_name: The model name.
        :param model_path: The path to save the model.
        :param data_path: The path to the dataset.
        :param know_distill: Whether the model uses knowledge distillation.
        :param max_p: The maximum p value.
        :param p_s: The p values to use for training.
        :param train_config: The training configuration.
        :param fjord_config: The configuration for Fjord.
        :param log_config: The logging configuration.
        :param device: The device to use.
        :param seed: The seed to use for the random number generator.
        """
        Logger.setup_logging(**log_config)
        self.cid = cid
        self.p_s = p_s
        self.net = get_net(model_name, p_s, device)
        self.trainloader, self.valloader = load_data(
            data_path, int(cid), train_config.batch_size, seed
        )

        self.know_distill = know_distill
        self.max_p = max_p
        self.fjord_config = fjord_config
        self.train_config = train_config
        self.model_path = model_path

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Get the parameters of the model to return to the server.

        :param config: The configuration.
        :return: The parameters of the model.
        """
        Logger.get().info(f"Getting parameters from client {self.cid}")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def net_to_state_dict_layers(self) -> List[Module]:
        """Model to state dict layers."""
        return net_to_state_dict_layers(self.net)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the parameters of the model.

        :param parameters: The parameters of the model.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[Tensor], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model on the training set.

        :param parameters: The parameters of the model.
        :param config: The train configuration.
        :return: The parameters of the model, the number of samples used for training,
            and the training metrics
        """
        Logger.get().info(
            f"Training on client {self.cid} for round "
            f"{config['current_round']!r}/{config['total_rounds']!r}"
        )

        original_parameters = deepcopy(parameters)

        self.set_parameters(parameters)
        self.train_config.lr = config["lr"]

        loss = train(
            self.net,
            self.trainloader,
            self.know_distill,
            self.max_p,
            p_s=self.p_s,
            epochs=self.train_config.local_epochs,
            current_round=int(config["current_round"]),
            total_rounds=int(config["total_rounds"]),
            train_config=self.train_config,
        )

        final_parameters = self.get_parameters(config={})

        return (
            final_parameters,
            len(self.trainloader.dataset),
            {
                "max_p": self.max_p,
                "p_s": self.p_s,
                "fjord_config": self.fjord_config,
                "original_parameters": original_parameters,
                "loss": loss,
            },
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Validate the model on the test set.

        :param parameters: The parameters of the model.
        :param config: The eval configuration.
        :return: The loss on the test set, the number of samples used for evaluation,
            and the evaluation metrics.
        """
        Logger.get().info(
            f"Evaluating on client {self.cid} for round "
            f"{config['current_round']!r}/{config['total_rounds']!r}"
        )

        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, [self.max_p])
        save_model(self.net, self.model_path, cid=self.cid)

        return loss[0], len(self.valloader.dataset), {"accuracy": accuracy[0]}
