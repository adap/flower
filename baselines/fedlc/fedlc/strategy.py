"""fedlc: A Flower Baseline."""
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
import re
from flwr.common.typing import UserConfig
import os
import torch
from flwr.server.client_manager import ClientManager
import numpy as np
from flwr.common import ndarrays_to_parameters
from fedlc.model import get_weights
from flwr.common.logger import log
from logging import INFO, DEBUG
import glob

# Adapted from:
#   https://flower.ai/docs/framework/how-to-save-and-load-model-checkpoints.html
class CheckpointedStrategyMixin(object):
    def __init__(self, net: torch.nn.Module, run_config: UserConfig):
        self.net = net
        self.strategy = str(run_config["strategy"])
        self.checkpoint_dir_path = str(run_config["checkpoint-dir-path"])
        self.save_params_every = int(run_config["save-params-every"])
        self.num_rounds = int(run_config["num-server-rounds"])
        self.dirichlet_alpha = float(run_config["dirichlet-alpha"])
        self.dataset = str(run_config["dataset"])
        self.last_round: int = 0
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        list_of_files = [fname for fname in glob.glob(f"{self.checkpoint_dir_path}/*.pth")]
        if list_of_files:
            latest_round_file = max(list_of_files, key=os.path.getctime)
            print(f"Loading pre-trained model from: {latest_round_file}")
            match = re.search(r"_r=(\d+)", latest_round_file)
            # Ensure checkpoints saved continue from last known round
            self.last_round = int(match.group(1)) if match else 0
            state_dict = torch.load(latest_round_file)
            self.net.load_state_dict(state_dict)
        else:
            print("No checkpoints found, using randomly initialized weights")

        weights = get_weights(self.net)
        initial_parameters = ndarrays_to_parameters(weights)
        return initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super(CheckpointedStrategyMixin, self).aggregate_fit( # type: ignore
            server_round, results, failures
        )

        if aggregated_parameters is not None and server_round % self.save_params_every == 0:
            # Create directory for saving model checkpoints
            os.makedirs(self.checkpoint_dir_path, exist_ok=True)

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            checkpoint_name: str = (
                f"{self.strategy}"
                f"_dataset={self.dataset.split('/')[-1]}"
                f"_R={self.num_rounds}"
                f"_alpha={self.dirichlet_alpha}"
                f"_r={self.last_round + server_round}"
            )
            checkpoint_file_path = os.path.join(self.checkpoint_dir_path,f"{checkpoint_name}.pth")
            torch.save(self.net.state_dict(), checkpoint_file_path)
            log(INFO, f"Saved checkpoint to {checkpoint_file_path} on round {server_round}")

        return aggregated_parameters, aggregated_metrics


class CheckpointedFedAvg(CheckpointedStrategyMixin, FedAvg):
    def __init__(self, net: torch.nn.Module, run_config: UserConfig, **kwargs):
        FedAvg.__init__(self, **kwargs)
        CheckpointedStrategyMixin.__init__(self, net, run_config)

class CheckpointedFedProx(CheckpointedStrategyMixin, FedProx):
    def __init__(self, net: torch.nn.Module, run_config: UserConfig, **kwargs):
        FedProx.__init__(self, **kwargs)
        CheckpointedStrategyMixin.__init__(self, net, run_config)