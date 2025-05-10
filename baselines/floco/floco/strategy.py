"""floco: A Flower Baseline."""

import copy
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from sklearn import decomposition

from flwr.common import (
    Context,
    EvaluateIns,
    FitIns,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    logger,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace


# pylint: disable=too-many-arguments
class CustomFedAvg(FedAvg):
    """Custom Federated Averaging strategy that stores and sends context object."""

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
                [int, NDArrays, Dict[str, Scalar], Context],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        context: Context,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        # Custom evaluation function that allows to send context object.
        self.eval_fn = evaluate_fn
        self.context = context

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.eval_fn(server_round, parameters_ndarrays, {}, self.context)
        if eval_res is None:
            return None

        return eval_res


class Floco(FedAvg):
    r"""Federated Optimization strategy.

    Implementation based on https://openreview.net/pdf?id=JL2eMCfDW8

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
    evaluate_fn: Optional[
        Callable[
            [int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]],
        ]
    ]
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
    tau: int = 0
        Round at which to start projection.
    rho: float = 1.0
        Radius of the ball around each projected client parameters
        from which models are sampled.
    endpoints: int = 1
        Number of endpoints of the solution simplex.
    """

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
                [int, NDArrays, Dict[str, Scalar], Context],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        context: Context,
        tau: int = 0,
        rho: float = 1.0,
        endpoints: int = 1,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.tau = tau
        self.rho = rho
        self.endpoints = endpoints
        self.last_selected_partition_ids: List[int] = []
        self.client_cid_to_partition_id: Dict = {}
        self.projected_clients: List = [ndarray]
        self.client_subregion_parameters: Dict = {}
        self.client_gradients: Dict = {}
        self.num_collected_client_gradients: int = 0
        self.context = context
        # Custom evaluation function that allows to send context object.
        self.eval_fn = evaluate_fn
        # Needed to compute pseudo gradients.
        self.initial_parameters: Parameters = initial_parameters

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.eval_fn(
            server_round,
            parameters_ndarrays,
            {
                "center": ndarray_to_bytes(
                    np.array([1 / self.endpoints for _ in range(self.endpoints)])
                ),
                "radius": self.rho,
                "endpoints": self.endpoints,
            },
            self.context,
        )
        if eval_res is None:
            return None

        return eval_res

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["server_round"] = server_round
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self.last_selected_partition_ids = [
            int(
                client.get_properties(
                    ins=GetPropertiesIns({}), group_id=server_round, timeout=30
                ).properties["partition-id"]
            )
            for client in clients
        ]
        if (server_round + 1) == self.tau:  # Round before projection
            regular_fraction_fit = self.fraction_fit
            self.fraction_fit = 1.0
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            self.num_collected_client_gradients = sample_size
            self.fraction_fit = regular_fraction_fit
            clients = client_manager.sample(  # Sample all clients to get gradients
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            # Create client cid to partition id mapping
            for client in clients:
                self.client_cid_to_partition_id[client.cid] = client.get_properties(
                    ins=GetPropertiesIns({}), group_id=server_round, timeout=30
                ).properties["partition-id"]

        elif server_round == self.tau:  # Round of projection
            # Get client gradients
            self.projected_clients = project_clients(
                self.client_gradients, self.endpoints
            )
            self.client_subregion_parameters = dict(
                zip(
                    np.arange(self.num_collected_client_gradients),
                    self.projected_clients,
                )
            )
        if server_round >= self.tau:
            fit_ins_all_clients = []
            for client in clients:
                tmp_client_partition_id = self.client_cid_to_partition_id[client.cid]
                tmp_client_config = copy.deepcopy(config)
                tmp_client_config["center"] = ndarray_to_bytes(
                    self.client_subregion_parameters[tmp_client_partition_id]
                )
                tmp_client_config["radius"] = self.rho
                tmp_fit_ins = FitIns(parameters, tmp_client_config)
                fit_ins_all_clients.append((client, tmp_fit_ins))
            return fit_ins_all_clients
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
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
        config["server_round"] = server_round
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round >= self.tau:
            eval_ins_all_clients = []
            for client in clients:
                tmp_client_partition_id = self.client_cid_to_partition_id[client.cid]
                tmp_client_config = copy.deepcopy(config)
                tmp_client_config["center"] = ndarray_to_bytes(
                    self.client_subregion_parameters[tmp_client_partition_id]
                )
                tmp_client_config["radius"] = self.rho
                tmp_eval_ins = EvaluateIns(parameters, tmp_client_config)
                eval_ins_all_clients.append((client, tmp_eval_ins))
            return eval_ins_all_clients

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        if (server_round + 1) == self.tau:  # All clients results are collected
            # Save client gradients for projection
            new_results = []
            for client, fit_res in results:
                tmp_client_partition_id = self.client_cid_to_partition_id[client.cid]
                w = parameters_to_ndarrays(fit_res.parameters)
                init_ndarrays = parameters_to_ndarrays(self.initial_parameters)
                client_grads = [
                    init_ndarrays[-i].flatten() - w[-i].flatten()
                    for i in range(1, self.endpoints + 1)
                ]  # Get pseudo gradients
                client_grads = np.concatenate(client_grads)
                self.client_gradients[tmp_client_partition_id] = client_grads
                self.client_gradients = {
                    k: self.client_gradients[k]
                    for k in sorted(self.client_gradients.keys())
                }
                if tmp_client_partition_id in self.last_selected_partition_ids:
                    # Only select the clients that were sampled
                    new_results.append((client, fit_res))
            results = new_results
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.initial_parameters = parameters_aggregated
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logger.log(WARNING, "No fit_metrics_aggregation_fn provided")
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        return loss, metrics


def project_clients(client_gradients, endpoints):
    """Optimize client projection onto a simplex of dimension endpoints-1."""
    client_stats = np.array(list(client_gradients.values()))
    kappas = decomposition.PCA(n_components=endpoints).fit_transform(client_stats)
    # Find optimal projection
    lowest_log_energy = np.inf
    best_beta = None
    for z in np.linspace(1e-4, 1, 1000):
        betas = _project_client_onto_simplex(kappas, z=z)
        betas /= betas.sum(axis=1, keepdims=True)
        log_energy = _riesz_s_energy(betas)
        if log_energy not in [-np.inf, np.inf] and log_energy < lowest_log_energy:
            lowest_log_energy = log_energy
            best_beta = betas
    return best_beta


def _project_client_onto_simplex(kappas, z):
    """Project clients onto a simplex of dimension endpoints-1."""
    sorted_kappas = np.sort(kappas, axis=1)[:, ::-1]
    z = np.ones(len(kappas)) * z
    cssv = np.cumsum(sorted_kappas, axis=1) - z[:, np.newaxis]
    ind = np.arange(kappas.shape[1]) + 1
    cond = sorted_kappas - cssv / ind > 0
    nonzero = np.count_nonzero(cond, axis=1)
    normalized_kappas = cssv[np.arange(len(kappas)), nonzero - 1] / nonzero
    betas = np.maximum(kappas - normalized_kappas[:, np.newaxis], 0)
    return betas


def _riesz_s_energy(simplex_points):
    """Compute Riesz s-energy of client projections.

    (https://www.sciencedirect.com/science/article/pii/S0021904503000315)
    """
    diff = simplex_points[:, None] - simplex_points[None, :]
    distance = np.sqrt((diff**2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    epsilon = 1e-4  # epsilon is the smallest distance possible to avoid overflow
    distance[distance < epsilon] = epsilon
    # select only upper triangular matrix to have each mutual distance once
    mutual_dist = distance[np.triu_indices(len(simplex_points), 1)]
    mutual_dist[np.argwhere(mutual_dist == 0).flatten()] = epsilon
    energies = 1 / mutual_dist**2
    energy = energies[~np.isnan(energies)].sum()
    log_energy = -np.log(len(mutual_dist)) + np.log(energy)
    return log_energy
