from typing import List

import os
import torch

from fjord.od.models import ResNet18
from fjord.utils.logger import Logger


def get_net(model_name: str, p_s: List[float],
            device: torch.device,
            ) -> torch.nn.Module:
    """
    Initialise model.
    :param model_name: name of the model
    :param p_s: list of p-values
    :param device: device to be used
    :return: initialised model"""
    if model_name == 'resnet18':
        net = ResNet18(
            od=True, p_s=p_s).to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return net


def save_model(model: torch.nn.Module, model_path: str,
               is_best: bool = False, cid: int = None
               ) -> None:
    """
    Checkpoint model.
    :param model: model to be saved
    :param model_path: path to save the model
    :param is_best: whether this is the best model
    :param cid: client id
    """
    suffix = "best" if is_best else "last"
    if cid:
        suffix += f"_{cid}"
    filename = os.path.join(model_path, f"model_{suffix}.checkpoint")
    Logger.get().info(f"Persisting model in {filename}")
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), filename)
