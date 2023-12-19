"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""
from flwr.server.server import fit_clients, Server
from flwr.server.history import History

import timeit
import numpy as np
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from flwr.common import (
    DisconnectRes,
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from .utils import (
    save_params,
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
            warmup_rounds: int,
            attack_fn:Callable,
            dataset_name: str,
            threshold: float = 0.0,
            to_keep: int = 1,
            magnitude: float = 0.0,
            sampling: int = 0,
            history_dir: str = "clients_params",
            omniscent: bool = True,
            *args: Any,
            **kwargs: Any
        ) -> None:
        """Initialize."""

        super().__init__(*args, **kwargs)
        self.num_malicious = num_malicious
        self.warmup_rounds = warmup_rounds
        self.attack_fn = attack_fn
        self.sampling = sampling
        self.aggregated_parameters = []
        self.params_indexes = []
        self.history_dir = history_dir
        self.dataset_name = dataset_name
        self.magnitude = magnitude
        self.malicious_selected = False
        self.threshold = threshold
        self.old_lambda = 0.0
        self.to_keep = to_keep
        self.omniscent = omniscent


    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        config = {"num_malicious": self.num_malicious, "attack_fn": self.attack_fn, "dataset_name": self.dataset_name}
        res = self.strategy.evaluate(0, parameters=self.parameters, config=config)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters, config=config)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    
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

        # Randomly decide which client is malicious
        size = self.num_malicious
        if self.warmup_rounds > server_round:
            size = 0
        print(f"fit_round - Selecting {size} malicious clients")
        self.malicious_lst = np.random.choice(
            [proxy.cid for proxy, _ in client_instructions], size=size, replace=False
        )
        log(
            DEBUG,
            "fit_round %s: malicious clients selected %s",
            server_round,
            self.malicious_lst,
        )
        # Save instruction for malicious clients into FitIns
        for proxy, ins in client_instructions:
            if proxy.cid in self.malicious_lst:
                ins.config["malicious"] = True
            else:
                ins.config["malicious"] = False

        for proxy, ins in client_instructions:
            print(f"fit_round - Client {proxy.cid} malicious: {ins.config['malicious']}")        
        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout
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

            print(f"fit_round 1 - Saving parameters of client {fitres.metrics['cid']} with shape {params.shape}")
            save_params(params, fitres.metrics['cid'], dir=self.history_dir)

            # Re-arrange results in the same order as clients' cids impose
            print("fit_round - Re-arranging results in the same order as clients' cids impose")
            ordered_results[int(fitres.metrics['cid'])] = (proxy, fitres)

        # Initialize aggregated_parameters if it is the first round
        if self.aggregated_parameters == []:
            for key, val in clients_state.items():
                if val == False:
                    self.aggregated_parameters = parameters_to_ndarrays(ordered_results[int(key)][1].parameters)
                    break

        # Apply attack function
        # the server simulates an attacker that controls a fraction of the clients
        if self.attack_fn is not None and server_round > self.warmup_rounds:
            print("fit_round - Applying attack function")
            results, others = self.attack_fn(
                ordered_results, clients_state, omniscent=self.omniscent, magnitude=self.magnitude,
                w_re=self.aggregated_parameters, malicious_selected=self.malicious_selected,
                threshold=self.threshold, d=len(self.aggregated_parameters), old_lambda=self.old_lambda,
                dataset_name=self.dataset_name, to_keep = self.to_keep,
                malicious_num=self.num_malicious
            )
            self.old_lambda = others.get('lambda', 0.0)

            # Update saved parameters time series after the attack
            for proxy, fitres in results:
                if fitres.metrics['malicious']:
                    if self.sampling > 0:
                        params = flatten_params(parameters_to_ndarrays(fitres.parameters))[self.params_indexes]
                    else:
                        params = flatten_params(parameters_to_ndarrays(fitres.parameters))
                    print(f"fit_round 2 - Saving parameters of client {fitres.metrics['cid']} with shape {params.shape}")
                    save_params(params, fitres.metrics['cid'], dir=self.history_dir, remove_last=True)
        else:
            results = ordered_results
            others = {}

        # Sort clients states
        clients_state = {k: clients_state[k] for k in sorted(clients_state)}
        print(f"fit_round - Clients state: {clients_state}")

        # Aggregate training results
        print("fit_round - Aggregating training results")
        aggregated_result = self.strategy.aggregate_fit(server_round, results, failures, clients_state)

        parameters_aggregated, metrics_aggregated, good_clients_idx, malicious_clients_idx = aggregated_result
        print(f"fit_round - Malicious clients: {malicious_clients_idx}")

        print(f"aggregate_fit - clients_state: {clients_state}")
        for idx in good_clients_idx:
            if clients_state[str(idx)]:
                self.malicious_selected = True
                break
            else:
                self.malicious_selected = False

        # For clients detected as malicious, set their parameters to be the averaged ones in their files
        # otherwise the forecasting in next round won't be reliable
        if self.warmup_rounds > server_round:
            print(f"fit_round - Saving parameters of clients")
            for idx in malicious_clients_idx:
                if self.sampling > 0:
                    new_params = flatten_params(parameters_to_ndarrays(parameters_aggregated))[self.params_indexes]
                else:
                    new_params = flatten_params(parameters_to_ndarrays(parameters_aggregated))
                save_params(new_params, idx, dir=self.history_dir, remove_last=True, rrl=True)

        return parameters_aggregated, metrics_aggregated, (results, failures)