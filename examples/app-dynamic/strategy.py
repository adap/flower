import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, Callable
from models import MNISTNet, CIFARNet
import torch

from flwr.common import (
    FitIns,
    Parameters,
    EvaluateIns,
    GetPropertiesIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays, NDArrays, MetricsAggregationFn,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

log = logging.getLogger(__name__)

CNFG_FIT = List[Tuple[ClientProxy, FitIns]]
CNFG_EVAL = List[Tuple[ClientProxy, EvaluateIns]]


def filter_by_task(
        workload: Union[CNFG_FIT, CNFG_EVAL], keyword: str
) -> Union[CNFG_FIT, CNFG_EVAL]:
    """This helper function filters the outputs of `configure_fit` and
    `configure_evaluate` and discards sending the instructions to those clients that
    aren't enrolled in a given task."""

    ins = GetPropertiesIns({})
    config_final = []
    for client, fit_or_eval_ins in workload:
        prop = client.get_properties(ins, timeout=30, group_id=None)
        if prop.properties[keyword]:
            config_final.append((client, fit_or_eval_ins))
            log.info(f"Client enrolled in {keyword} federation.")
        else:
            log.info(f"Client excluded from {keyword} federation.")

    return config_final


class CustomFedAvg(FedAvg):
    """FedAvg for task specific training."""

    def __init__(self, *, dataset_name: str, net_name: str, fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2, min_available_clients: int = 2,
                 evaluate_fn: Optional[
                     Callable[
                         [int, NDArrays, Dict[str, Scalar]],
                         Optional[Tuple[float, Dict[str, Scalar]]],
                     ]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[
                     Callable[[int], Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None,
                 fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 inplace: bool = True) -> None:
        if initial_parameters is None:
            initial_parameters = self._get_initial_parameters(net_name)
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients,
                         min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients,
                         evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn,
                         on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures,
                         initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         inplace=inplace)
        self.dataset_name = dataset_name
        self.net_name = net_name

    def _get_initial_parameters(self, task):
        """Initiate parameters."""
        if task == "mnist_net":
            model = MNISTNet()
        elif task == "cifar_net":
            model = CIFARNet()
        else:
            raise ValueError("The task must be mnist or cifar")
        ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: ClientManager
    ) -> CNFG_FIT:
        """Configure fit that filters (clients, instructions) based on task."""
        # Copy the original configure_fit to modify the config
        config = {}
        config["dataset"] = self.dataset_name
        config["net"] = self.net_name

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        fit_config = [(client, fit_ins) for client in clients]

        # Now discard based on a task
        return filter_by_task(fit_config, self.dataset_name)

    def configure_evaluate(
            self, server_round: int, parameters: Parameters,
            client_manager: ClientManager
    ) -> CNFG_EVAL:
        """Configure evaluate that filters (clients, instructions) based on task."""
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        config["dataset"] = self.dataset_name
        config["net"] = self.net_name
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        eval_config = [(client, evaluate_ins) for client in clients]

        # Now discard based on client's enrollment status
        return filter_by_task(eval_config, self.dataset_name)

    def evaluate(self, server_round: int, parameters: Parameters):
        # Call evaluate as it would normally be called. We will return
        # it's output so to not interrupt the normal flow
        evaluate_output = super().evaluate(server_round, parameters)
        # Additionally, here saving of the model can happen
        return evaluate_output
