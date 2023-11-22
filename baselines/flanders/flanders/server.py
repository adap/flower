"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""
from flwr.server import Server

import concurrent.futures
import timeit
import numpy as np
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    parameters_to_ndarrays
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from .utils import (
    save_params, 
    save_predicted_params,
    load_all_time_series, 
    load_time_series, 
    flatten_params
)

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

class EnhancedServer(Server):
    """Server with enhanced functionality."""

    def __init__(
            self,
            num_malicious: int,
            attack_fn:Optional[Callable],
            *args: Any,
            **kwargs: Any
        ) -> None:
        """Initialize."""

        super().__init__(*args, **kwargs)

    
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated learning."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        # Randomly decide which client is malicious
        if server_round > self.warmup_rounds:
            self.malicious_selected = np.random.choice(
                [proxy.cid for proxy, ins in client_instructions], size=self.num_malicious, replace=False
            )
            log(
                DEBUG,
                "fit_round %s: malicious clients selected %s",
                server_round,
                self.malicious_selected,
            )
            # Save instruction for malicious clients into FitIns
            for proxy, ins in client_instructions:
                if proxy.cid in self.malicious_selected:
                    ins["malicious"] = True
                else:
                    ins["malicious"] = False

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
        results, failures = super.fit_clients(
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

        clients_state = {}      # dictionary of clients representing wether they are malicious or not

        # Save parameters of each client as time series
        ordered_results = [0 for _ in range(len(results))]
        cids = np.array([])
        for proxy, fitres in results:
            cids = np.append(cids, int(fitres.metrics["cid"]))
            clients_state[fitres.metrics['cid']] = fitres.metrics['malicious']
            params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            if self.sampling > 0:
                if len(self.params_indexes) == 0:
                    # Sample a random subset of parameters
                    self.params_indexes = np.random.randint(0, len(params), size=self.sampling)

                params = params[self.params_indexes]

            save_params(params, fitres.metrics['cid'])

            # Re-arrange results in the same order as clients' cids impose
            ordered_results[int(fitres.metrics['cid'])] = (proxy, fitres)

        # Initialize aggregated_parameters if it is the first round
        if self.aggregated_parameters == []:
            for key, val in clients_state.items():
                if val == False:
                    self.aggregated_parameters = parameters_to_ndarrays(ordered_results[int(key)][1].parameters)
                    break

        # Apply attack function
        if self.attack_fn is not None and server_round > self.warmup_rounds:
            results, others = self.attack_fn(
                ordered_results, clients_state, magnitude=self.magnitude,
                w_re=self.aggregated_parameters, malicious_selected=self.malicious_selected,
                threshold=self.threshold, d=len(self.aggregated_parameters), old_lambda=self.old_lambda,
                dataset_name=self.dataset_name, agr_function=self.strategy_name, to_keep = self.to_keep,
                malicious_num=self.m[-1]
            )
            self.old_lambda = others.get('lambda', 0.0)

            # Update saved parameters time series after the attack
            for proxy, fitres in results:
                if fitres.metrics['malicious']:
                    if self.sampling > 0:
                        params = flatten_params(parameters_to_ndarrays(fitres.parameters))[self.params_indexes]
                    else:
                        params = flatten_params(parameters_to_ndarrays(fitres.parameters))
                    save_params(params, fitres.metrics['cid'], remove_last=True)
        else:
            results = ordered_results
            others = {}

        # Sort clients states
        clients_state = {k: clients_state[k] for k in sorted(clients_state)}

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures, clients_state)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
