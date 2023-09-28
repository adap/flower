"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

# Flower server

import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor

from hfedxgboost.models import CNN
from hfedxgboost.utils import Early_Stop, single_tree_preds_from_each_client, test

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
        self,
        *,
        client_manager: ClientManager,
        early_stopper: Early_Stop,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self._client_manager = client_manager
        self.parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
        self.strategy = strategy
        self.max_workers = None
        self.early_stopper = early_stopper

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
        """Run a round of fitting on the server side.

        Parmeters:
            self: The instance of the class.
            server_round (int): The round of server communication.
            timeout (float, optional): Maximum time to wait for client responses.

        Returns
        -------
            The aggregated CNN model and tree
            Aggregated metric value
            A tuple containing the results and failures.

            None if no clients were selected.
        """
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
        CNN_aggregated, trees_aggregated = aggregated[0], aggregated[1]

        if isinstance(trees_aggregated, list):
            print("Server side aggregated", len(trees_aggregated), "trees.")
        else:
            print("Server side did not aggregate trees.")

        return (
            [CNN_aggregated, trees_aggregated],
            metrics_aggregated,
            (results, failures),
        )

    def check_res_cen(self, current_round, timeout, start_time, history):
        """Get results after fitting the model for the current round.

        Check if those results are not None and check
        if the training should stop or not.

        Parameters
        ----------
            current_round (int): The current round number.
            timeout (int): The time limit for the evaluation.
            start_time (float): The starting time of the evaluation.
            history (History): The object for storing the evaluation history.

        Returns
        -------
            bool: True if the early stop criteria is met, False otherwise.
        """
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
            if self.early_stopper.early_stop(res_cen):
                return True
        return False

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated learning for a given number of rounds.

        Parameters
        ----------
            num_rounds (int): The number of rounds to run federated learning.
            timeout (Optional[float]): The optional timeout value for each round.

        Returns
        -------
            History: The history object that stores the loss and metrics data.
        """
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
        for current_round in range(1, num_rounds + 1):
            stop = self.check_res_cen(current_round, timeout, start_time, history)
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
            # Stop if no progress is happenning
            if stop:
                break

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
        """Validate current global model on a number of clients.

        Parameters
        ----------
            server_round (int): representing the current server round
            timeout (float, optional): The time limit for the request in seconds.

        Returns
        -------
            Aggregated loss,
            Aggregated metric,
            Tuple of the results and failures.
            or
            None if no clients selected.
        """
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
        aggregated_result = self.strategy.aggregate_evaluate(
            server_round, results, failures
        )

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def _get_initial_parameters(
        self, timeout: Optional[float]
    ) -> Tuple[Parameters, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        """Get initial parameters from one of the available clients.

        Parmeters:
            timeout (float, optional): The time limit for the request in seconds.
            Defaults to None.

        Returns
        -------
            parameters (tuple): A tuple containing the initial parameters.
        """
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
    """Perform server-side evaluation.

    Parameters
    ----------
        server_round (int): The round of server evaluation.
        parameters (Tuple): A tuple containing the parameters needed for evaluation.
            First element: an instance of the Parameters class.
            Second element: a tuple consists of either an XGBClassifier
            or XGBRegressor model and an integer, or a list of that tuple.
        config (Dict): A dictionary containing configuration parameters.
        cfg: Hydra configuration object.
        testloader (DataLoader): The data loader used for testing.
        batch_size (int): The batch size used for testing.

    Returns
    -------
        Tuple[float, Dict]: A tuple containing the evaluation loss (float) and
        a dictionary containing the evaluation metric(s) (float).
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_name = cfg.dataset.task.metric.name

    device = cfg.device
    model = CNN(cfg)
    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]
    testloader = single_tree_preds_from_each_client(
        testloader,
        batch_size,
        trees_aggregated,
        cfg.n_estimators_client,
        cfg.client_num,
    )
    loss, result, _ = test(cfg, model, testloader, device=device, log_progress=False)

    print(
        f"Evaluation on the server: test_loss={loss:.4f},",
        f"test_,{metric_name},={result:.4f}",
    )
    return loss, {metric_name: result}
