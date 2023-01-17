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

from collections import defaultdict
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import datetime
import numpy as np
import numpy.typing as npt
import ray
import pickle
from time import sleep, time_ns
from pathlib import Path, PurePath
from ray.experimental.state.api import list_actors
from numpy.linalg import lstsq

from flwr.common import (
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
from flwr.monitoring.profiler import (
    SimpleCPU,
    SimpleCPUProcess,
    SimpleGPU,
    SimpleGPUProcess,
    Task,
)

from flwr.server.client_manager import ClientManager

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# pylint: disable=too-many-locals

MAX_POLY_DEG: int = 15
MAX_INT_64 = 9223372036854775807


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
        resource_poly_degree: int = 1,  # Rename this
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
        monitor_namespace: str = "flwr_experiment",
        profiles: Dict[
            str, Dict[str, int]
        ] = {},  # Eventually, change this to List[Profiles]
        num_warmup_steps: int = 100,
        save_models_folder: Path = Path("/home/pedro/flwr_monitor/"),
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
        self.current_round: int = 0
        self.client_configs_map: Dict[
            str, Tuple[str, str, int]
        ] = {}  # {client_id:(node_id, uuid, num_steps)}
        self.namespace = monitor_namespace
        self.profiles = profiles
        self.resource_poly_degree: int = resource_poly_degree
        self.cpu_resources: Dict[str, SimpleCPU] = {}  # node_id: simple_cpu
        self.gpu_resources: Dict[
            str, Dict[str, SimpleGPU]
        ] = {}  # node_id: {gpu_uuid:  simple_gpu}
        self.resources_model: Dict[
            Tuple[str, str], Tuple[np.ndarray, int]
        ] = {}  # {(node_id, gpu_uuid): (Polynomial, max_num_clients)
        self.resource_model_x = defaultdict(list)
        self.resource_model_y = defaultdict(list)
        self.start_time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.num_warmup_steps = num_warmup_steps
        self.begin_round: int = 0
        self.save_models_folder = save_models_folder

    def __repr__(self) -> str:
        rep = f"ResourceAwareFedAvg(accept_failures={self.accept_failures})"
        return rep

    def _get_monitors(self):
        actors = list_actors(
            filters=[
                ("state", "=", "ALIVE"),
                ("class_name", "=", "RaySystemMonitor"),
            ]
        )
        return actors

    def _start_data_collection(self):
        actors = self._get_monitors()
        for actor in actors:
            node_id: str = actor["name"]
            this_actor = ray.get_actor(node_id, namespace=self.namespace)
            this_actor.run.remote()

    def _stop_data_collection(self):
        actors = self._get_monitors()
        for actor in actors:
            node_id: str = actor["name"]
            this_actor = ray.get_actor(node_id, namespace=self.namespace)
            ray.get(this_actor.stop.remote())

    def _save_and_clear_monitors(
        self,
        *,
        server_round: int = 0,
        parameters: Optional[Parameters],
        sub_dir: Optional[PurePath] = None,
    ):
        # This is suitable for a single FL training. If multiple, then each
        # monitor should be doing its thing
        sub_dir = (
            sub_dir
            if sub_dir
            else PurePath(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        )

        actors = self._get_monitors()
        for actor in actors:
            node_id: str = actor["name"]
            this_actor = ray.get_actor(node_id, namespace=self.namespace)
            ray.get(this_actor.save_and_clear.remote(sub_folder=sub_dir))

        # save_model
        if parameters:
            tmp_dir = self.save_models_folder / self.start_time
            tmp_dir.mkdir(exist_ok=True, parents=True)
            with open(tmp_dir / f"{str(server_round)}.pickle", "wb") as f:
                pickle.dump(parameters, f)

    def request_available_resources(self) -> None:
        actors = self._get_monitors()
        for actor in actors:
            node_id: str = actor["name"]
            this_actor = ray.get_actor(node_id, namespace=self.namespace)

            # Get resources
            obj_ref = this_actor.get_resources.remote()
            this_node_resources = cast(
                Dict[str, Union[SimpleCPU, SimpleGPU]], ray.get(obj_ref)
            )
            # Create initial model for each GPU
            self.gpu_resources[node_id] = {}
            for k, v in this_node_resources.items():
                if isinstance(v, SimpleGPU):  # Only get GPUs for now
                    self.gpu_resources[node_id][k] = v
                    gpu_uuid = v.uuid
                    self.create_gpu_poly_model(node_id=node_id, gpu_uuid=gpu_uuid)
                elif isinstance(v, SimpleCPU):
                    self.cpu_resources[node_id] = v

    def create_gpu_poly_model(self, node_id: str, gpu_uuid: str):
        poly_model = np.array([1.0, 1.0, 1.0])
        self.resources_model[(node_id, gpu_uuid)] = poly_model, 1

    def generate_client_priority(
        self, clients: List[ClientProxy], properties: Dict[str, Union[float, int]]
    ):  # This entire function should be passed to the strategy as a parameters
        weighted_clients = [(x, properties[x.cid]) for x in clients]
        return weighted_clients

    def configure_warmup_round(
        self,
        *,
        client_manager: ClientManager,
        parameters: Parameters,
        num_clients_per_gpu: int,
        config: Dict[str, Scalar],
    ) -> List[Tuple[ClientProxy, FitIns]]:

        clients: List[ClientProxy] = client_manager.sample(
            num_clients=num_clients_per_gpu * len(self.resources_model),
            min_num_clients=num_clients_per_gpu * len(self.resources_model),
        )

        client_fit_list: List[Tuple[ClientProxy, FitIns]] = []

        for (node_id, gpu_uuid), (_, max_num_clients) in self.resources_model.items():
            gpu_id = self.gpu_resources[node_id][gpu_uuid].gpu_id

            _, max_num_clients = self.resources_model[(node_id, gpu_uuid)]
            these_clients = clients[: min(max_num_clients, len(clients))]

            for client in these_clients:
                # Ray node allocation
                client.resources["num_cpus"] = 1
                client.resources["resources"] = {gpu_uuid: 1.0 / max_num_clients}
                num_steps = (
                    np.ceil(
                        self.profiles["train"][client.cid] // int(config["batch_size"])
                    )
                    * config["epochs"]
                )
                this_config = dict(
                    config,
                    **{"gpu_id": gpu_id, "local_steps": self.num_warmup_steps},
                )
                self.client_configs_map[client.cid] = (
                    node_id,
                    gpu_uuid,
                    num_steps,
                )
                client_fit_list.append((client, FitIns(parameters, this_config)))
        return client_fit_list

    def aggregate_warmup_round_time(self, results: List[Tuple[ClientProxy, FitRes]]):
        task_to_cid: Dict[Scalar, str] = {}
        for client, result in results:
            this_task_id = result.metrics["_flwr.monitoring.task_id"]
            task_to_cid[this_task_id] = client.cid

        actors = self._get_monitors()
        for actor in actors:
            node_id: str = actor["name"]
            this_monitor = ray.get_actor(node_id, namespace=self.namespace)
            obj_ref = this_monitor.aggregate_statistics.remote(
                task_ids=[k for k in task_to_cid.keys()]
            )
            this_monitor_metrics = ray.get(obj_ref)

            # Find earliest Task starting time and latest
            # Task finishing time per GPU.
            starting_times: Dict[Tuple[str, str], int] = {}
            finishing_times: Dict[Tuple[str, str], int] = {}
            total_num_steps: Dict[Tuple[str, str], int] = {}
            num_clients: Dict[Tuple[str, str], int] = {}

            for task_id, times_ns in this_monitor_metrics["training_times_ns"].items():
                cid = task_to_cid[task_id]
                node_id, gpu_uuid, num_steps = self.client_configs_map[cid]

                c_start_time, c_finish_time = times_ns
                p_start_time = starting_times.get((node_id, gpu_uuid), MAX_INT_64)
                p_finish_time = finishing_times.get((node_id, gpu_uuid), 0)

                starting_times[(node_id, gpu_uuid)] = min(p_start_time, c_start_time)
                finishing_times[(node_id, gpu_uuid)] = max(p_finish_time, c_finish_time)

                # Accumulate both time and number of clients
                total_num_steps[(node_id, gpu_uuid)] = (
                    total_num_steps.get((node_id, gpu_uuid), 0) + num_steps
                )
                num_clients[(node_id, gpu_uuid)] = (
                    num_clients.get((node_id, gpu_uuid), 0) + 1
                )

            # Save GPU training duration variables
            for node_id, gpu_uuid in self.resources_model.keys():
                x_1 = num_clients[(node_id, gpu_uuid)]
                x_2 = total_num_steps[(node_id, gpu_uuid)]
                self.resource_model_x[(node_id, gpu_uuid)].append((1, x_1, x_2))
                self.resource_model_y[(node_id, gpu_uuid)].append(
                    (
                        finishing_times[(node_id, gpu_uuid)]
                        - starting_times[(node_id, gpu_uuid)]
                    )
                    / 1e9
                )

    def associate_resources(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
        clients_with_weights: List[Tuple[ClientProxy, int]],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients_with_weights.sort(key=lambda x: x[1], reverse=True)

        expected_training_times: List[float] = len(self.resources_model) * [0.0]
        node_gpu_mapping = [k for k in self.resources_model.keys()]

        client_fit_list: List[Tuple[ClientProxy, FitIns]] = []
        while clients_with_weights:
            # Choose which GPU to use, based on time
            idx = expected_training_times.index(min(expected_training_times))
            node_id, gpu_uuid = node_gpu_mapping[idx]
            gpu_id = self.gpu_resources[node_id][gpu_uuid].gpu_id

            this_config = dict(
                config,
                **{
                    "gpu_id": gpu_id,
                },
            )

            # Associate multiple clients to a single GPU if possible (maximize VRAM utilization)
            model_poly, max_num_clients = self.resources_model[(node_id, gpu_uuid)]
            actual_num_clients = min(max_num_clients, len(clients_with_weights))

            these_clients = clients_with_weights[:actual_num_clients]
            sum_local_steps = np.sum([x[1] for x in these_clients])

            # consider largest num_steps
            multi_client_per_gpu_expected_time = np.matmul(
                model_poly, [1, actual_num_clients, sum_local_steps]
            )

            expected_training_times[idx] += multi_client_per_gpu_expected_time

            # Now associate resources
            for client, num_samples in these_clients:
                client.resources["resources"] = {gpu_uuid: 1.0 / actual_num_clients}
                client.resources["num_cpus"] = 1
                num_steps = np.ceil(int(num_samples) / int(config["batch_size"]))
                self.client_configs_map[client.cid] = (node_id, gpu_uuid, num_steps)
                client_fit_list.append((client, FitIns(parameters, this_config)))

            del clients_with_weights[:actual_num_clients]

        print(f"Expected training time: {max(expected_training_times)} seconds.")

        return client_fit_list

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configures the next round of training and allocates resources accordingly."""

        self._start_data_collection()

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        client_fit_list: List[Tuple[ClientProxy, FitIns]] = []

        if server_round == 1:
            # Calculate maximum number of clients per GPU
            # based on VRAM
            self.request_available_resources()

            client_fit_list = self.configure_warmup_round(
                client_manager=client_manager,
                parameters=parameters,
                num_clients_per_gpu=1,
                config=config,
            )

        elif server_round == 2:
            client_fit_list = self.configure_warmup_round(
                client_manager=client_manager,
                parameters=parameters,
                num_clients_per_gpu=2,
                config=config,
            )

        elif server_round == 3:
            client_fit_list = self.configure_warmup_round(
                client_manager=client_manager,
                parameters=parameters,
                num_clients_per_gpu=3,
                config=config,
            )

        else:  # All other rounds
            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            clients_with_weights = [
                (client, self.profiles["train"][client.cid]) for client in clients
            ]

            # Return client/config pairs
            client_fit_list = self.associate_resources(
                parameters, config, clients_with_weights
            )
        self.begin_round = time_ns()
        return client_fit_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        self._stop_data_collection()
        # Print training time
        print(f"Effective training time: {(time_ns()-self.begin_round)/1e9}")

        # Task_id used in this round
        round_tasks_to_cid: Dict[Scalar, str] = {}
        for client, fit_res in results:
            this_task_id = fit_res.metrics["_flwr.monitoring.task_id"]
            round_tasks_to_cid[this_task_id] = client.cid

        if server_round == 1:
            # Calculate maximum number of clients per GPU
            # And accumulate time spent for single user, different num steps
            # Create client-task_id mapping

            actors = self._get_monitors()
            for actor in actors:
                node_id: str = actor["name"]
                this_monitor = ray.get_actor(node_id, namespace=self.namespace)
                obj_ref = this_monitor.aggregate_statistics.remote(
                    task_ids=[k for k in round_tasks_to_cid.keys()]
                )
                this_monitor_metrics = ray.get(obj_ref)

                # Get GPU memory usage. Here we consider single gpu per task,
                # But we should also consider all possible combinations of multiple GPUs as well.
                max_this_proc_mem_used_mb: Dict[
                    str, Dict[str, float]  # {task_id:{gpu_uuid:float}}
                ] = this_monitor_metrics["max_this_proc_mem_used_mb"]
                max_all_proc_mem_used_mb = this_monitor_metrics[
                    "max_all_proc_mem_used_mb"
                ]
                # Find maximum number of clients, considering the number of CPUs
                total_num_available_cpu_cores = self.cpu_resources[node_id].num_cores
                cores_per_gpu = total_num_available_cpu_cores // len(
                    self.resources_model
                )
                print(f"Maximum number of cores per GPU: {cores_per_gpu}")

                # Calculate maximum memory
                for this_task, gpu_mem_dict in max_this_proc_mem_used_mb.items():
                    # Just one task per gpu, as scheduled
                    for (gpu_uuid, max_mem_this_proc_this_gpu) in gpu_mem_dict.items():
                        total_mem_this_gpu = self.gpu_resources[node_id][
                            gpu_uuid
                        ].total_mem_mb
                        max_all_proc_mem = max_all_proc_mem_used_mb[gpu_uuid]
                        print(
                            f"Total Memory for GPU: {gpu_uuid} = {total_mem_this_gpu}"
                        )
                        print(f"Max all proc mem GPU: {gpu_uuid} = {max_all_proc_mem}")
                        print(
                            f"Max mem this process GPU: {gpu_uuid} {max_mem_this_proc_this_gpu}"
                        )

                        # Try and consider the system memory as well
                        vram_max_num_clients_this_gpu = int(
                            (
                                total_mem_this_gpu
                                - max_all_proc_mem
                                + max_mem_this_proc_this_gpu
                            )
                            / max_mem_this_proc_this_gpu
                        )
                        # max_num_clients_this_gpu = vram_max_num_clients_this_gpu
                        max_num_clients_this_gpu = min(
                            vram_max_num_clients_this_gpu, cores_per_gpu
                        )

                        # Update the resource_model max number of clients per gpu
                        poly_model, t = self.resources_model[(node_id, gpu_uuid)]
                        self.resources_model[(node_id, gpu_uuid)] = (
                            poly_model,
                            max_num_clients_this_gpu,
                        )
                        print(
                            f"Maximum number of clients for {node_id}, {gpu_uuid} = {max_num_clients_this_gpu}"
                        )
                        if vram_max_num_clients_this_gpu > cores_per_gpu:
                            print("Limited by CPU cores.")

                # Include time for training single user
                # t = t_0 + t_1*N + t_2*(sum) per GPU, here N = 1
                task_id_to_cid: Dict[Scalar, str] = {}
                for client, result in results:
                    this_task_id = result.metrics["_flwr.monitoring.task_id"]
                    task_id_to_cid[this_task_id] = client.cid

                for task_id, training_time_ns in this_monitor_metrics[
                    "training_times_ns"
                ].items():
                    cid = task_id_to_cid[task_id]
                    node_id, gpu_uuid, num_steps = self.client_configs_map[cid]
                    self.resource_model_x[(node_id, gpu_uuid)].append((1, 1, num_steps))
                    self.resource_model_y[(node_id, gpu_uuid)].append(
                        (training_time_ns[1] - training_time_ns[0]) / 1e9
                    )

        elif server_round == 2:
            self.aggregate_warmup_round_time(results)

        elif server_round == 3:
            self.aggregate_warmup_round_time(results)

            # fit Polynomials
            for (
                node_id,
                gpu_uuid,
            ), (_, max_num_clients) in self.resources_model.items():
                x = self.resource_model_x[(node_id, gpu_uuid)]
                y = self.resource_model_y[(node_id, gpu_uuid)]

                poly_model, res, rank, s = lstsq(x, y, rcond=None)

                self.resources_model[(node_id, gpu_uuid)] = (
                    poly_model,
                    max_num_clients,
                )
        # Tell monitor to store and continue
        agg_results = super().aggregate_fit(server_round, results, failures)
        sub_dir = PurePath(self.start_time, str(server_round))
        self._save_and_clear_monitors(
            sub_dir=sub_dir, parameters=agg_results[0], server_round=server_round
        )

        return agg_results
