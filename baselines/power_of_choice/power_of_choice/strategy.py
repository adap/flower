"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import EvaluateIns, FitIns, GetPropertiesIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy.fedavg import WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW, FedAvg
from flwr.common.logger import log
import math
import numpy as np

class SimpleCriterion(Criterion):
    def __init__(self, clients_cid: List[str]) -> None:
        self.clients_cid = clients_cid

    def select(self, client: ClientProxy) -> bool:
        return client.cid in self.clients_cid


class PowerOfChoice(FedAvg):
    "Custom strategy."

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        d: int,
        ck: int,
        variant: Optional[str] = "base",
        atmp: Optional[Dict[str, float]] = None
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`,
            `min_evaluate_clients` will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        d : int
            Candidate Client set size.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.d = d
        self.ck = ck
        self.res_first_phase = None
        self.candidate_set_clients = None
        self.selected_clients_criterion = None
        self.variant = variant
        self.atmp = atmp

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Compute sample size m
        sample_size = max(self.ck, 1)

        log(
            INFO,
            f"Server {server_round}: about to sample {sample_size} clients"
        )

        if self.variant == "base":
            # Assumption is that all the clients in d have returned a result from the first phase
            # Sort the list based on the loss attribute of EvaluateRes in descending order
            sorted_client_losses = sorted(self.res_first_phase, key=lambda x: x[1].loss, reverse=True)

            # Take the top m clients from the sorted list
            chosen_clients = [client[0] for client in sorted_client_losses[:sample_size]]

            # Build criterion
            self.selected_clients_criterion = SimpleCriterion(list(map(lambda x: x.cid, chosen_clients)))
        else:
            # In case of the rpow variant, we need to sample the d clients from the Amax set
            # Sort the atmp dictionary based on the value in descending order
            sorted_atmp = sorted(self.atmp.items(), key=lambda x: x[1], reverse=True)

            # Filter out the clients that are not in the candidate set
            sorted_atmp = list(filter(lambda x: x[0] in self.candidate_set_clients, sorted_atmp))

            # Take the top m clients from the sorted list
            chosen_clients = [client[0] for client in sorted_atmp[:sample_size]]

            # Build criterion
            self.selected_clients_criterion = SimpleCriterion(list(chosen_clients))
        
        # Sample clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=sample_size,
            criterion=self.selected_clients_criterion,
        )

        log(
            INFO,
            f"Round {server_round}: selected clients {list(map(lambda x: x.cid, clients))} for fit.",
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager, first_phase: bool = False
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # Set first_phase flag
        config["first_phase"] = first_phase
        
        evaluate_ins = EvaluateIns(parameters, config)

        if first_phase:
            # Sample d clients from the available clients
            sample_size = self.sample_clients(client_manager)

            criterion = SimpleCriterion(self.candidate_set_clients)

            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=sample_size,
                criterion=criterion
            )

            log(
                INFO,
                f"Server {server_round}: selected clients {list(map(lambda x: x.cid, clients))} for first phase evaluation.",
            )

        else:

            # Compute sample size m
            sample_size = max(self.ck, 1)
            
            # Sample clients
            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=sample_size,
                criterion=self.selected_clients_criterion,
            )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def set_res_first_phase(self, res_first_phase):
        self.res_first_phase = res_first_phase

    def sample_clients(self, client_manager: ClientManager) -> int:
        """Sample d clients from the client manager, returns the number of clients sampled."""
        num_available_clients = client_manager.num_available()
        sample_size = min(self.d, num_available_clients)

        available_clients = client_manager.all()

        data_samples_per_client = {}

        # Pass an empty config dictionary, no need to tell the clients any config
        config = {}

        # Get the data size of each client
        for cid, available_client in available_clients.items():
            propertiesRes = available_client.get_properties(GetPropertiesIns(config), None)
            data_size = propertiesRes.properties["data_size"]
            data_samples_per_client[cid] = data_size

        # Extract a subset of t clients, where each client has a probability of being extracted proportional to its data size
        client_ids = list(data_samples_per_client.keys())
        client_probabilities = [data_samples_per_client[cid] for cid in client_ids]
        client_probabilities_normalized = [p / sum(client_probabilities) for p in client_probabilities]

        self.candidate_set_clients = np.random.choice(
            client_ids,
            size=sample_size,
            replace=False,
            p=client_probabilities_normalized,
        )

        return sample_size

    def update_atmp(self, res_eval):
        """Update the atmp dictionary with the loss of the clients in res_eval"""
        for result in res_eval:
            self.atmp[result[0].cid] = result[1].loss