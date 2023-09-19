"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""
client_num=5
task_type="REG"

# Flower server
import functools
from flwr.server.strategy import FedXgbNnAvg
from flwr.server.app import ServerConfig

import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import  Parameters, Scalar, EvaluateRes,FitRes,parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor
from .models import CNN
from .utils import single_tree_preds_from_each_client,test
from .client import FL_Client
from flwr.server.server import (
    fit_clients,
    evaluate_clients,
)
from hydra.utils import instantiate

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]


class FL_Server(fl.server.Server):
    """Flower server."""

    def __init__(
        self,*, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager = client_manager
        self.parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy = strategy
        self.max_workers = None

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)

        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
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
        best_res=-99999999999999
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  
                if parameters_prime:
                    self.parameters = parameters_prime
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
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
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
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

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[
            Optional[
                Tuple[
                    Parameters,
                    Union[
                        Tuple[XGBClassifier, int],
                        Tuple[XGBRegressor, int],
                        List[
                            Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
                        ],
                    ],
                ]
            ],
            Dict[str, Scalar],
            FitResultsAndFailures,
        ]
    ]:
        """Perform a single round of federated averaging."""

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

        metrics_aggregated: Dict[str, Scalar]
        aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        NN_aggregated, trees_aggregated = aggregated[0], aggregated[1]

        if type(trees_aggregated) is list:
            print("Server side aggregated", len(trees_aggregated), "trees.")
        else:
            print("Server side did not aggregate trees.")

        return (
            [NN_aggregated, trees_aggregated],
            metrics_aggregated,
            (results, failures),
        )

    def _get_initial_parameters(
        self, timeout: Optional[float]
    ) -> Tuple[Parameters, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        """Get initial parameters from one of the available clients."""
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res_tree = random_client.get_parameters(ins=ins, timeout=timeout)
        parameters = [get_parameters_res_tree[0].parameters, get_parameters_res_tree[1]]
        log(INFO, "Received initial parameters from one random client")

        return parameters

def serverside_eval(
    server_round: int,
    parameters: Tuple[
        Parameters,
        Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ],
    ],
    config: Dict[str, Scalar],
    cfg,
    testloader: DataLoader,
    batch_size: int,
) -> Tuple[float, Dict[str, float]]:
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_name=cfg.dataset.metric.name
    
    device = cfg.device
    model = CNN(cfg)
    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]
    testloader = single_tree_preds_from_each_client(
        testloader, batch_size, trees_aggregated, cfg.dataset.n_estimators_client,cfg.client_num
    )
    loss, result, _ = test(
        cfg.dataset.task_type, model, testloader, device=device, log_progress=False
    )

    print(f"Evaluation on the server: test_loss={loss:.4f}, test_,{metric_name},={result:.4f}")
    return loss, {metric_name: result}


