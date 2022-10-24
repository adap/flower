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

from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
from ray.experimental.state.api import list_actors
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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

# from flwr.common.logger import log
# from flwr.common.typing import Config
from flwr.monitoring.profiler import SimpleGPUProcess

from flwr.server.client_manager import ClientManager

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


# pylint: disable=too-many-locals


class ResourceAwareFedAvg(FedAvg):
    """Configurable ResourceAwareFedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
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
        monitor_namespace: str = "raysysmon",
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        # will store the updated resources for clients in the round
        self.updated_resources_for_clients = []
        self.namespace: str = monitor_namespace
        self.available_gpus: Dict[str, Dict[str, SimpleGPUProcess]] = {}

    def __repr__(self) -> str:
        rep = f"ResourceAwareFedAvg(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        client_fit_list: List[Tuple[ClientProxy, FitIns]] = []

        # first round serves to collect data
        if server_round == 1:
            actors = list_actors(
                filters=[
                    ("state", "=", "ALIVE"),
                    ("class_name", "=", "RaySystemMonitor"),
                ]
            )
            num_gpus = 0
            for actor in actors:
                node_id = actor["name"]
                this_actor = ray.get_actor(actor[node_id], namespace=self.namespace)
                obj_ref = this_actor.get_resources.remote()
                gpus_in_this_node = obj_ref.get()
                self.available_gpus[node_id] = gpus_in_this_node
                num_gpus = num_gpus + len(gpus_in_this_node)

            # Sample one client per GPU. This is an first approximation.
            # We need to understand how the GPUs behave using more than
            # one client at a time
            clients: List[ClientProxy] = client_manager.sample(
                num_clients=num_gpus, min_num_clients=num_gpus
            )

            # Create individual configs that will be sent to each client
            list_fitins: List[FitIns] = []
            list_resources: List[Dict[str, Any]] = []
            for node_id in self.available_gpus.keys():
                for gpu in self.available_gpus[node_id].values():
                    list_fitins.append(
                        FitIns(
                            parameters,
                            {
                                **config,
                                "_flwr.gpu_id": gpu.id,
                            },
                        )
                    )
                    node_aff = NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                    list_resources.append({"scheduling_strategy": node_aff})

            for idx, resource in enumerate(list_resources):
                clients[idx].resources = resource

            client_fit_list = list(zip(clients, list_fitins))

        else:
            # continue as usual
            fit_ins = FitIns(parameters, config)

            # Based on existing statistics, choose right amount of resources
            # This could begin with an initial guess.

            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # update resources for clients
            # iterate over data from clients in the previous round
            for idx, resource in enumerate(list_resources):
                clients[idx].resources = {"num_gpu": 0.5}  # resource

            # Return client/config pairs
            client_fit_list = [(client, fit_ins) for client in clients]
        return client_fit_list


'''
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # Collect Statistics
        actors = list_actors(
            filters=[("state", "=", "ALIVE"), ("class_name", "=", "RaySystemMonitor")]
        )
        collected_stats = {}
        for actor in actors:
            ray_sysmon = ray.get_actor(actor["name"])
            obj_ref = ray_sysmon.aggregate_statistics.remote()
            collected_stats = {
                **collected_stats,
                **obj_ref.get(),
            }
        print(collected_stats)

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

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
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
'''
