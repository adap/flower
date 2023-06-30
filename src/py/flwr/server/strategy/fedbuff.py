# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Learning with Buffered Asynchronous Aggregation (FedBuff)

[Nguyen et al., 2021] strategy.

Paper:

https://arxiv.org/abs/2106.06639
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.server.criterion import Criterion

from .aggregate import aggregate
from .fedavg import FedAvg


# flake8: noqa: E501
class FedBuff(FedAvg):
    """Configurable FedBuff strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        concurrency: int = 3,
        buffer_size: int = 2,
        staleness_fn: Optional[Callable[[int], int]] = None,
    ) -> None:
        """Federated Buffering asynchronous aggregation strategy.
        NOTE: requires server to be in asynchronous mode
        Implementation based on https://arxiv.org/abs/2106.06639


        Parameters
        ----------
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        concurrency : int
            Number of clients that should be training at any one time. Note, additional clients are only added after
            aggregation rounds, so there may be periods with fewer clients in use.
        buffer_size : int
            Number of client updates to collect before running aggregation.
        staleness_fn : Optional[Callable[[int], float]]
            Function that takes the age of an update in aggregation rounds (where 0 is no missed rounds) and outputs
            a multiplier used to scale the gradients from that update. Defaults to returning 1.0 for all inputs.
        """
        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.current_params_ndarray: Optional[NDArrays] = None

        self.concurrency = concurrency
        self.buffer_size = buffer_size
        self.staleness_fn: Callable[[int], int]

        if staleness_fn is None:
            self.staleness_fn = lambda x: 1
        else:
            self.staleness_fn = staleness_fn

        self.busy_clients: Dict[str, int] = {}  # dict from cid to server round

    def __repr__(self) -> str:
        rep = f"FedBuff(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        _ = num_available_clients
        num_additional_clients = self.concurrency - len(self.busy_clients)
        return num_additional_clients, num_additional_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["buffer_size"] = self.buffer_size
        fit_ins = FitIns(parameters, config)

        # Save so can work out gradients
        self.current_params_ndarray = parameters_to_ndarrays(parameters)

        print("Working out which clients to instruct")

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        print(f"Want {sample_size} more clients, minimum {min_num_clients}")

        occupied_clients = self.busy_clients.keys()

        class NotBusyCriterion(Criterion):
            """Criterion to select only non busy clients."""

            def select(self, client: ClientProxy) -> bool:
                return client.cid not in occupied_clients

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=NotBusyCriterion(),
        )
        print(f"Selected clients = {[c.cid for c in clients]}")

        self.busy_clients.update({c.cid: server_round for c in clients})

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # No federated evaluation
        return []

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

        results_cids = [result[0].cid for result in results]
        failures_cids = [
            failure[0].cid
            for failure in failures
            if not isinstance(failure, BaseException)
        ]

        # How many rounds off each result is
        staleness = [server_round - self.busy_clients[c] for c in results_cids]

        print(f"These clients sent updates: {list(zip(results_cids,staleness))}")
        for cid in results_cids:
            self.busy_clients.pop(cid)

        print(f"These clients failed: {failures_cids}")
        for cid in failures_cids:
            self.busy_clients.pop(cid)

        if self.current_params_ndarray is None:
            raise ValueError("Current parameters NDArray is None")

        # Convert results to list of (delta,weight) tuples
        deltas_results: List[Tuple[NDArrays, int]] = [
            (
                # Need to do each layer separately as may be different sizes
                [
                    new_layer_params - current_layer_params
                    for new_layer_params, current_layer_params in zip(
                        parameters_to_ndarrays(fit_res.parameters),
                        self.current_params_ndarray,
                    )
                ],
                # Weight by num_examples scaled according to staleness
                self.staleness_fn(age) * fit_res.num_examples,
            )
            for age, (_, fit_res) in zip(staleness, results)
        ]

        # Need to add each layer delta seperately
        self.current_params_ndarray = [
            layer_delta + current_layer_params
            for layer_delta, current_layer_params in zip(
                aggregate(deltas_results),
                self.current_params_ndarray,
            )
        ]

        parameters_aggregated = ndarrays_to_parameters(self.current_params_ndarray)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # No federated evaluation
        return None, {}
