"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import List, Tuple, Optional, Union, Dict, NDArrays
from omegaconf import DictConfig
from flwr.common.logger import log
from logging import DEBUG, INFO
from flwr.common import Code, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from scaffold.strategy import FitIns, FitRes, FitResultsAndFailures

import torch
import concurrent.futures
from hydra.utils import instantiate

class ScaffoldServer(Server):
    """Flower server implementing the communication for scaffold."""
    
    def __init__(self,
        strategy: Strategy,
        model: DictConfig,
        client_manager: ClientManager=None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.server_cv = self._init_control_variates(model)
        self.model_params = instantiate(model)
    
    def _init_control_variates(self, model:DictConfig):
        net = instantiate(model)
        server_cv = []
        for param in net.parameters():
            server_cv.append(torch.zeros(param.shape))
        return server_cv

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
            server_cv=self.server_cv,
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

        # Collect `fit` results from all clients participating in this round
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
            Optional[NDArrays],
            Dict[str, Scalar],
            Optional[NDArrays]
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        aggregated_cv_update = aggregated_result[2]
        self.server_cv = [torch.from_numpy(cv + cv_multiplier*aggregated_cv_update[i]) for i, cv in enumerate(server_cv_np)]

        # update parameters x = x + 1* aggregated_update
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [x + aggregated_result[0][i] for i, x in enumerate(curr_params)]
        parameters_aggregated = ndarrays_to_parameters(updated_params)
        
        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_aggregated, metrics_aggregated, (results, failures)

def fit_clients(
        client_instructions: List[Tuple[ClientProxy, FitIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
    ) -> FitResultsAndFailures:
        """Refine parameters concurrently on all selected clients."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            submitted_fs = {
                executor.submit(fit_client, client_proxy, ins, timeout)
                for client_proxy, ins in client_instructions
            }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # Gather results
        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_fit(
                future=future, results=results, failures=failures
            )
        return results, failures

def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)