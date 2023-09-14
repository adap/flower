"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moon.models import init_net, test


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
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
        net = init_net(cfg.dataset.name, cfg.model.name, cfg.model.output_dim)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        accuracy, loss = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
