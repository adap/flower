"""fedlc: A Flower Baseline."""

import os
import re
from collections import OrderedDict
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import flwr as fl
from fedlc.model import get_parameters
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx

from .utils import gen_checkpoint_suffix


# Adapted from:
#   https://flower.ai/docs/framework/how-to-save-and-load-model-checkpoints.html
class CheckpointedStrategyMixin(object):
    def __init__(
        self, net: torch.nn.Module, run_config: UserConfig, use_last_checkpoint: bool
    ):
        self.net = net
        self.strategy = str(run_config["strategy"])
        self.checkpoint_dir_path = str(run_config["checkpoint-dir-path"])
        self.save_params_every = int(run_config["save-params-every"])
        self.num_rounds = int(run_config["num-server-rounds"])
        self.dirichlet_alpha = float(run_config["dirichlet-alpha"])
        self.use_last_checkpoint = use_last_checkpoint
        self.dataset = str(run_config["dataset"])
        self.last_round: int = 0
        if use_last_checkpoint:
            # Create directory for saving model checkpoints
            os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        else:
            log(
                INFO,
                "Not loading from previous checkpoint. Set use-last-checkpoint=True to enable.",
            )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.use_last_checkpoint:
            checkpoint_suffix = gen_checkpoint_suffix(
                self.strategy,
                self.dataset,
                self.dirichlet_alpha,
            )
            checkpoints_list = [
                f"{self.checkpoint_dir_path}/{f}"
                for f in os.listdir(self.checkpoint_dir_path)
                if checkpoint_suffix in f
            ]
            if checkpoints_list:
                latest_round_file = max(checkpoints_list, key=os.path.getctime)
                log(INFO, f"Loading pre-trained model from: {latest_round_file}")
                match = re.search(r"_r=(\d+)", latest_round_file)
                # Ensure checkpoints saved continue from last known round
                self.last_round = int(match.group(1)) if match else 0
                state_dict = torch.load(latest_round_file)
                self.net.load_state_dict(state_dict)
            else:
                log(INFO, "No checkpoints found, using randomly initialized weights")

        weights = get_parameters(self.net)
        initial_parameters = ndarrays_to_parameters(weights)
        return initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super(CheckpointedStrategyMixin, self).aggregate_fit(  # type: ignore
            server_round, results, failures
        )

        if (
            self.use_last_checkpoint
            and aggregated_parameters is not None
            and server_round % self.save_params_every == 0
        ):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            checkpoint_suffix = gen_checkpoint_suffix(
                self.strategy,
                self.dataset,
                self.dirichlet_alpha,
            )
            checkpoint_name = f"{checkpoint_suffix}{self.last_round+server_round}.pth"
            checkpoint_file_path = os.path.join(
                self.checkpoint_dir_path, checkpoint_name
            )
            torch.save(self.net.state_dict(), checkpoint_file_path)
            log(
                INFO,
                f"Saved checkpoint to {checkpoint_file_path} on round {server_round}",
            )

        return aggregated_parameters, aggregated_metrics


class CheckpointedFedAvg(CheckpointedStrategyMixin, FedAvg):
    def __init__(
        self,
        net: torch.nn.Module,
        run_config: UserConfig,
        use_last_checkpoint: bool,
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        CheckpointedStrategyMixin.__init__(self, net, run_config, use_last_checkpoint)


class CheckpointedFedProx(CheckpointedStrategyMixin, FedProx):
    def __init__(
        self,
        net: torch.nn.Module,
        run_config: UserConfig,
        use_last_checkpoint: bool,
        **kwargs,
    ):
        FedProx.__init__(self, **kwargs)
        CheckpointedStrategyMixin.__init__(self, net, run_config, use_last_checkpoint)
