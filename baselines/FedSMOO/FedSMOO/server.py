"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.typing import Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
from flwr.common.typing import NDArray
from flwr.server.history import History

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
from flwr.common.logger import log
from logging import DEBUG, INFO


import torch
import torch.nn as nn
from flwr.common.typing import NDArrays, Scalar
from flwr.common.parameter import *
from flwr.server.server import *    # import server and utils
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from FedSMOO.models import test, set_H_param_list

# Flower Server class needed!! 
# need to store/ aggregate more than just the parameters

class FedSMOOServer(Server):
    """ Flower FedSMOO server with an additional 
    set of parameters for global perturbations """

    def __init__(
        self,
        net: nn.Module,
        sam_lr: float,
        client_manager: ClientManager,
        strategy: Strategy = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )

        self.net = net  # to store the network as a template
        self.sam_lr = sam_lr
        self.strategy = strategy if strategy is not None else FedAvg()

        self.init_mdl_param = [val.cpu().numpy() for _, val in self.net.state_dict().items()] # list of np arrays
        self.gs_diff_list = [torch.from_numpy(t) for t in self.init_mdl_param] # list of torch tensors
        self.max_workers: Optional[int] = None

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        print(f"\n------------------------- Round {server_round} ---------------------------\n")

        config = {"gs_diff_list": [mat.numpy() for mat in self.gs_diff_list],
                  "init_mdl_param": self.init_mdl_param,}
        
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            config = config,
            client_manager=self._client_manager,)

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        # results -> returned by client fit (params, len dataloader, dict)
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            NDArray,
        ] = self.strategy.aggregate_fit(server_round, self.sam_lr, results, failures)

        parameters_aggregated, gs_normalized = aggregated_result
        self.gs_diff_list = set_H_param_list(self.net, gs_normalized)
        return parameters_aggregated, {}, (results, failures) # results and failures are not used by super.fit()


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
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
