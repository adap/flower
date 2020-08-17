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
"""Federating: Fast and Slow (v1)."""


from logging import DEBUG, INFO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .fast_and_slow import (
    is_fast_round,
    next_timeout,
    normalize_and_sample,
    timeout_candidates,
)
from .fedavg import FedAvg

E = 0.001
E_TIMEOUT = 0.0001
WAIT_TIMEOUT = 600


class FedFSv1(FedAvg):
    """Strategy implementation which alternates between sampling fast and slow
    cients."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        min_completion_rate_fit: float = 0.5,
        min_completion_rate_evaluate: float = 0.5,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        dynamic_timeout_percentile: float = 0.8,
        r_fast: int = 1,
        r_slow: int = 1,
        t_max: int = 10,
        use_past_contributions: bool = False,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        self.min_completion_rate_fit = min_completion_rate_fit
        self.min_completion_rate_evaluate = min_completion_rate_evaluate
        self.dynamic_timeout_percentile = dynamic_timeout_percentile
        self.r_fast = r_fast
        self.r_slow = r_slow
        self.t_max = t_max
        self.use_past_contributions = use_past_contributions
        self.contributions: Dict[str, List[Tuple[int, int, int]]] = {}
        self.durations: List[Tuple[str, float, int, int]] = []

    def __repr__(self) -> str:
        # pylint: disable-msg=line-too-long
        rep = f"FedFSv1(dynamic_timeout_percentile={self.dynamic_timeout_percentile}, r_fast={self.r_fast}, r_slow={self.r_slow}, t_max={self.t_max})"
        return rep

    # pylint: disable-msg=too-many-locals
    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Block until `min_num_clients` are available
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        success = client_manager.wait_for(
            num_clients=min_num_clients, timeout=WAIT_TIMEOUT
        )
        if not success:
            # Do not continue if not enough clients are available
            log(
                INFO,
                "FedFS: not enough clients available after timeout %s",
                WAIT_TIMEOUT,
            )
            return []

        # Sample clients
        if rnd == 1:
            # Sample with 1/k in the first round
            log(
                DEBUG,
                "FedFS round %s, sample %s clients with 1/k",
                str(rnd),
                str(sample_size),
            )
            clients = self._one_over_k_sampling(
                sample_size=sample_size, client_manager=client_manager
            )
        else:
            fast_round = is_fast_round(rnd - 1, r_fast=self.r_fast, r_slow=self.r_slow)
            log(
                DEBUG,
                "FedFS round %s, sample %s clients, fast_round %s",
                str(rnd),
                str(sample_size),
                str(fast_round),
            )
            clients = self._fs_based_sampling(
                sample_size=sample_size,
                client_manager=client_manager,
                fast_round=fast_round,
            )

        # Prepare parameters and config
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Use custom fit config function if provided
            config = self.on_fit_config_fn(rnd)

        # Set timeout for this round
        if self.durations:
            candidates = timeout_candidates(
                durations=self.durations, max_timeout=self.t_max,
            )
            timeout = next_timeout(
                candidates=candidates, percentile=self.dynamic_timeout_percentile,
            )
            config["timeout"] = str(timeout)
        else:
            # Initial round has not past durations, use max_timeout
            config["timeout"] = str(self.t_max)

        # Fit instructions
        fit_ins = (parameters, config)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def _one_over_k_sampling(
        self, sample_size: int, client_manager: ClientManager
    ) -> List[ClientProxy]:
        """Sample clients with probability 1/k."""
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return clients

    def _fs_based_sampling(
        self, sample_size: int, client_manager: ClientManager, fast_round: bool
    ) -> List[ClientProxy]:
        """Sample clients with 1/k * c/m in fast rounds and 1 - c/m in slow rounds."""
        all_clients: Dict[str, ClientProxy] = client_manager.all()
        k = len(all_clients)
        cid_idx: Dict[int, str] = {}
        raw: List[float] = []
        for idx, (cid, _) in enumerate(all_clients.items()):
            cid_idx[idx] = cid

            if cid in self.contributions.keys():
                # Previously selected clients
                contribs: List[Tuple[int, int, int]] = self.contributions[cid]

                # pylint: disable-msg=invalid-name
                if self.use_past_contributions:
                    cs = [c for _, c, _ in contribs]
                    ms = [m for _, _, m in contribs]
                    c_over_m = sum(cs) / sum(ms)
                else:
                    _, c, m = contribs[-1]
                    c_over_m = c / m
                # pylint: enable-msg=invalid-name

                if fast_round:
                    importance = (1 / k) * c_over_m + E
                else:
                    importance = 1 - c_over_m + E
            else:
                # Previously unselected clients
                if fast_round:
                    importance = 1 / k
                else:
                    importance = 1
            raw.append(importance)

        log(
            DEBUG,
            "FedFS _fs_based_sampling, sample %s clients, raw %s",
            str(sample_size),
            str(raw),
        )

        return normalize_and_sample(
            all_clients=all_clients,
            cid_idx=cid_idx,
            raw=np.array(raw),
            sample_size=sample_size,
            use_softmax=False,
        )

    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None

        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.min_completion_rate_fit:
            # Not enough results for aggregation
            return None

        # Convert results
        weights_results = [
            (parameters_to_weights(parameters), num_examples)
            for client, (parameters, num_examples, _, _) in results
        ]
        weights_prime = aggregate(weights_results)

        # Track contributions to the global model
        for client, fit_res in results:
            cid = client.cid
            contribution: Tuple[int, int, int] = (rnd, fit_res[1], fit_res[2])
            if cid not in self.contributions.keys():
                self.contributions[cid] = []
            self.contributions[cid].append(contribution)

        self.durations = []
        for client, (_, num_examples, num_examples_ceil, fit_duration) in results:
            cid_duration = (
                client.cid,
                fit_duration,
                num_examples,
                num_examples_ceil,
            )
            self.durations.append(cid_duration)

        return weights_prime

    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.min_completion_rate_evaluate:
            # Not enough results for aggregation
            return None

        return weighted_loss_avg([evaluate_res for _, evaluate_res in results])
