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
"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: https://arxiv.org/pdf/1710.06963.pdf
"""


from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPFixedClipStrategy(Strategy):
    """Wrapper for configuring a Strategy for DP with Fixed Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        num_sampled_clients: int,
        clip_norm: float,
        noise_multiplier: float = 1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        # Doing fixed-size subsampling as in https://arxiv.org/abs/1905.03871.
        self.num_sampled_clients = num_sampled_clients
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def __repr__(self) -> str:
        rep = "Strategy with DP with Fixed Clipping enabled."
        return rep

    # Instead of adding noise with std dev sigma = z*clip_norm at server,
    # add noise with std dev sigma_dash = z*clip_norm/sqrt(m) at
    # each of the m chosen clients.
    def _calc_client_noise_stddev(self) -> float:
        return float(
            self.noise_multiplier * self.clip_norm / (self.num_sampled_clients ** (0.5))
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        client_instructions = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )

        noise_stddev = self._calc_client_noise_stddev()
        for _, fit_ins in client_instructions:
            fit_ins.config["clip_norm"] = self.clip_norm
            fit_ins.config["noise_stddev"] = noise_stddev

        return client_instructions

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1

        return self.strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)
