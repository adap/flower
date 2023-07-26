"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

import torch
import random

from typing import Callable, Dict, Optional, Tuple
from threading import Thread
from omegaconf import DictConfig
from collections import OrderedDict
from hydra.utils import instantiate
from FedPer.models import test
from torch.utils.data import DataLoader
from flwr.common.typing import NDArrays, Scalar

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""

        net = instantiate(model)
        model_keys = [k for k in net.state_dict().keys() if k.startswith("body")]
        model_keys = [k.replace("body.", "") for k in model_keys]
        params_dict = zip(model_keys, parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.body.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate