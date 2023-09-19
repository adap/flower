from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from models import test
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import time


def gen_evaluate_fn(
    trainloader : DataLoader,
    testloader: DataLoader,
    device: torch.device,
    model,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    trainloader: DataLoader
        The Dataloader that was used for training the main model
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
        """Use the entire test set for evaluation."""

        net = model
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)


        print('in stat')
        start_time = time.time()
        with torch.no_grad():
            net.train(True)
            for i, input in enumerate(trainloader):
                input_dict = {}
                input_dict['img'] = input[0].to(device)
                input_dict['label'] = input[1].to(device)
                net(input_dict)
        end_time = time.time()
        print(f'end of stat, time taken = {start_time - end_time}')


        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        print(f'schezwan accuracy = {accuracy}')
        return loss, {"accuracy": accuracy}

    return evaluate
