import datetime
import json
import random
import pickle
import numpy as np
from copy import deepcopy
from pathlib import Path
from flwr.server.strategy import Strategy
from typing import Dict, List, Optional, Tuple, Union
from flwr.common.typing import GetPropertiesIns, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from math import ceil
from collections import defaultdict


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)


class MultiNodeWrapper(Strategy):
    def __init__(
        self,
        num_virtual_clients_fit_per_round: int,
        num_virtual_clients_fit_total: int,
        num_virtual_clients_eval_total: int,
        num_virtual_clients_eval_per_round: int,
        aggregating_strategy: Strategy,
        path_save_model: Optional[Path] = None,
    ) -> None:
        self.num_virtual_clients_fit_per_round = num_virtual_clients_fit_per_round
        self.num_virtual_clients_eval_per_round = num_virtual_clients_eval_per_round
        self.num_virtual_clients_fit_total = num_virtual_clients_fit_total
        self.num_virtual_clients_eval_total = num_virtual_clients_eval_total
        self.aggregating_strategy = aggregating_strategy
        self.map_node_list_gpu_id: Dict[str, List[str]] = defaultdict(list)
        self.map_node_list_cid: Dict[str, List[str]] = defaultdict(list)
        self.map_cid_gpu_id: Dict[str, str] = {}
        self.path_save_model = Path(path_save_model) if path_save_model else None
        self.starting_time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def __repr__(self) -> str:
        rep = f"Multi-Node version of {self.aggregating_strategy}"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:

        return self.aggregating_strategy.initialize_parameters(client_manager)

    def gen_cid_gpu_ids_map(self):
        for node_name, workers_cid_this_node in self.map_node_list_cid.items():
            this_node_gpus = self.map_node_list_gpu_id[node_name]
            for idx, cid in enumerate(workers_cid_this_node):
                self.map_cid_gpu_id[cid] = this_node_gpus[idx % len(this_node_gpus)]

    def get_node_to_cid_and_gpu_maps(
        self, list_config_fits: List[Tuple[ClientProxy, FitIns]]
    ):
        ins = GetPropertiesIns(config={})
        for client_proxy, _ in list_config_fits:
            cid = client_proxy.cid
            worker_properties = client_proxy.get_properties(
                ins=ins, timeout=60
            ).properties
            node_name = str(worker_properties["node_name"])
            self.map_node_list_cid[node_name].append(cid)

            # Workers from same node will report same GPUs
            if node_name not in self.map_node_list_gpu_id.keys():
                # keys relative to a GPU in properties begin with GPU
                all_gpus_uuids = [
                    gpu_uuid
                    for gpu_uuid in worker_properties.keys()
                    if gpu_uuid.startswith("GPU")
                ]

                # Now load all GPUS to get their IDs
                for gpu_uuid in all_gpus_uuids:
                    gpu_dict = json.loads(str(worker_properties[gpu_uuid]))
                    gpu_id = gpu_dict["id"]
                    self.map_node_list_gpu_id[node_name].append(gpu_id)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        list_config_fits = self.aggregating_strategy.configure_fit(
            server_round, parameters, client_manager
        )
        num_workers = len(list_config_fits)

        ## Generate random selection of virtual clients (number of virtual clients per round)
        sampled_virtual_cids = random.sample(
            range(self.num_virtual_clients_fit_total),
            self.num_virtual_clients_fit_per_round,
        )

        # chunk_size = ceil(len(sampled_virtual_cids) / len(list_config_fits))
        # p_list = [
        #    sampled_virtual_cids[i : i + chunk_size]
        #    for i in range(0, len(sampled_virtual_cids), chunk_size)
        # ]

        ## Get clients properties. This should be done elsewhere at the beginning by the Server+ClientManager:
        if server_round == 1:
            # Get {node_name:[cids]} and {node_name:[gpu_ids]}
            self.get_node_to_cid_and_gpu_maps(list_config_fits)

            # Get {cid:gpu_ids}
            self.gen_cid_gpu_ids_map()
        else:
            pass

        ## Round-Robin:
        # Creates clients splits amongst workers
        p_list = np.array_split(sampled_virtual_cids, num_workers)

        for idx, (c, f) in enumerate(list_config_fits):
            a = deepcopy(f)
            a.config["list_clients"] = "_".join(
                [str(virtual_cid) for virtual_cid in p_list[idx]]
            )
            a.config["gpu_id"] = self.map_cid_gpu_id[c.cid]
            list_config_fits[idx] = (c, a)

        # Return client/config pairs
        return list_config_fits

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        agg_results = self.aggregating_strategy.aggregate_fit(
            server_round, results, failures
        )
        if self.path_save_model:
            path_and_timestamp = self.path_save_model / self.starting_time
            path_and_timestamp.mkdir(parents=True, exist_ok=True)
            with open(path_and_timestamp / f"{server_round}.pickle", "wb") as f:
                pickle.dump(agg_results[0], f)

        return agg_results

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.aggregating_strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.aggregating_strategy.aggregate_evaluate(
            server_round, results, failures
        )

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.aggregating_strategy.evaluate(server_round, parameters)
