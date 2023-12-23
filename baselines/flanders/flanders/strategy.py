"""FLANDERS strategy."""

import typing
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from .utils import load_all_time_series

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class Flanders:
    """Aggregation function based on MAR.

    Take a look at the paper for more details about the parameters.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, too-many-locals
    def __init__(
        self,
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
        to_keep: int = 1,
        window: int = 0,
        maxiter: int = 100,
        alpha: float = 1,
        beta: float = 1,
        distance_function=None,
    ) -> None:
        """Initialize FLANDERS.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during the fit phase, by default 1.0
        fraction_evaluate : float, optional
            Fraction of clients used during the evaluate phase, by default 1.0
        min_fit_clients : int, optional
            Minimum number of clients used during the fit phase, by default 2
        min_evaluate_clients : int, optional
            Minimum number of clients used during the evaluate phase, by
            default 2
        min_available_clients : int, optional
            Minimum number of clients available for training and evaluation, by
            default 2
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
        Optional[Tuple[float, Dict[str, Scalar]]]]], optional
            Evaluation function, by default None
        on_fit_config_fn : Optional[Callable[[int], Dict[str, Scalar]]],
        optional
            Function to generate the config fed to the clients during the fit
            phase, by default None
        on_evaluate_config_fn : Optional[Callable[[int], Dict[str, Scalar]]],
        optional
            Function to generate the config fed to the clients during the
            evaluate phase, by default None
        accept_failures : bool, optional
            Whether to accept failures from clients, by default True
        initial_parameters : Optional[Parameters], optional
            Initial model parameters, by default None
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn], optional
            Function to aggregate metrics during the fit phase, by default None
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn],
        optional
            Function to aggregate metrics during the evaluate phase, by default
            None
        to_keep : int, optional
            Number of clients to keep (i.e., to classify as "good"), by default
            1
        window : int, optional
            Sliding window size used as a "training set" of MAR, by default 0
        maxiter : int, optional
            Maximum number of iterations of MAR, by default 100
        alpha : float, optional
            Alpha parameter (regularization), by default 1
        beta : float, optional
            Beta parameter (regularization), by default 1
        distance_function : Callable, optional
            Distance function used to compute the distance between predicted
            params and real ones, by default None
        """
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
        self.to_keep = to_keep
        self.window = window
        self.maxiter = maxiter
        self.alpha = alpha
        self.beta = beta
        self.params_indexes = None
        self.malicious_selected = False
        self.distance_function = distance_function

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FLANDERS(accept_failures={self.accept_failures})"
        return rep

    @typing.no_type_check
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    @typing.no_type_check
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    # pylint: disable=unused-argument
    @typing.no_type_check
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    @typing.no_type_check
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # Custom FitIns object for each client
        fit_ins_list = [
            FitIns(
                parameters,
                {}
                if not self.on_fit_config_fn
                else self.on_fit_config_fn(server_round),
            )
            for _ in range(sample_size)
        ]

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        result = []
        for client, fit in zip(clients, fit_ins_list):
            result.append((client, fit))
        return result

    # pylint: disable=too-many-locals
    @typing.no_type_check
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        clients_state: Dict[int, bool],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Apply MAR forecasting to exclude malicious clients from FedAvg.

        Parameters
        ----------
        server_round : int
            Current server round.
        results : List[Tuple[ClientProxy, FitRes]]
            List of results from the clients.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            List of failures from the clients.

        Returns
        -------
        parameters_aggregated: Optional[Parameters]
            Aggregated parameters.
        metrics_aggregated: Dict[str, Scalar]
            Aggregated metrics.
        malicious_clients_idx: List[int]
            List of malicious clients' cids (indexes).
        """
        good_clients_idx = []
        malicious_clients_idx = []
        if server_round > 1:
            win = self.window
            if server_round < self.window:
                win = server_round
            params_tensor = load_all_time_series(
                params_dir="clients_params", window=win
            )
            params_tensor = np.transpose(
                params_tensor, (0, 2, 1)
            )  # (clients, params, time)

            ground_truth = params_tensor[:, :, -1].copy()
            pred_step = 1
            log(INFO, "Computing MAR on params_tensor %s", params_tensor.shape)
            predicted_matrix = mar(
                params_tensor[:, :, :-1],
                pred_step,
                maxiter=self.maxiter,
                alpha=self.alpha,
                beta=self.beta,
            )

            log(INFO, "Computing anomaly scores")
            anomaly_scores = self.distance_function(
                ground_truth, predicted_matrix[:, :, 0]
            )
            log(INFO, "Anomaly scores: %s", anomaly_scores)

            log(INFO, "Selecting good clients")
            good_clients_idx = sorted(
                np.argsort(anomaly_scores)[: self.to_keep]
            )  # noqa
            malicious_clients_idx = sorted(
                np.argsort(anomaly_scores)[self.to_keep :]
            )  # noqa
            results = np.array(results)[good_clients_idx].tolist()
            log(INFO, "Good clients: %s", good_clients_idx)

        log(INFO, "Applying FedAvg")
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            parameters_aggregated,
            metrics_aggregated,
            good_clients_idx,
            malicious_clients_idx,
        )

    @typing.no_type_check
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
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
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar],
        output_dir: str,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters.

        Evaluate model parameters using an evaluation function (centralized evaluation).
        """
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        eval_res = self.evaluate_fn(
            server_round, parameters_to_ndarrays(parameters), config, output_dir
        )  # type: ignore [call-arg]
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if (not results) or (not self.accept_failures and failures):
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


# pylint: disable=too-many-locals, too-many-arguments, invalid-name
def mar(X, pred_step, alpha=1, beta=1, maxiter=100, window=0):
    """Forecast the next tensor of params.

    Forecast the next tensor of params by using MAR algorithm.

    Code provided by Xinyu Chen at:
    https://towardsdatascience.com/ matrix-autoregressive-model-for-multidimensional-
    time-series-forecasting-6a4d7dce5143

    With some modifications.
    """
    m, n, T = X.shape
    start = 0
    if window > 0:
        start = T - window

    A = np.random.randn(m, m)
    B = np.random.randn(n, n)
    X_norm = (X - np.min(X)) / np.max(X)

    for _ in range(maxiter):
        temp0 = B.T @ B
        temp1 = np.zeros((m, m))
        temp2 = np.zeros((m, m))
        identity_m = np.identity(m)

        for t in range(start, T):
            temp1 += X_norm[:, :, t] @ B @ X_norm[:, :, t - 1].T
            temp2 += X_norm[:, :, t - 1] @ temp0 @ X_norm[:, :, t - 1].T

        temp2 += alpha * identity_m
        A = temp1 @ np.linalg.inv(temp2)

        temp0 = A.T @ A
        temp1 = np.zeros((n, n))
        temp2 = np.zeros((n, n))
        identity_n = np.identity(n)

        for t in range(start, T):
            temp1 += X_norm[:, :, t].T @ A @ X_norm[:, :, t - 1]
            temp2 += X_norm[:, :, t - 1].T @ temp0 @ X_norm[:, :, t - 1]

        temp2 += beta * identity_n
        B = temp1 @ np.linalg.inv(temp2)

    tensor = np.append(X, np.zeros((m, n, pred_step)), axis=2)
    for s in range(pred_step):
        tensor[:, :, T + s] = A @ tensor[:, :, T + s - 1] @ B.T
    return tensor[:, :, -pred_step:]
