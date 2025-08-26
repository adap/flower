# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Message-based Central differential privacy with fixed clipping.

Papers: https://arxiv.org/abs/1712.07557, https://arxiv.org/abs/1710.06963
"""


from logging import INFO, WARNING
from typing import Optional

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.common.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_clip_model_update,
    compute_stdv,
)
from flwr.common.differential_privacy_constants import CLIENTS_DISCREPANCY_WARNING
from flwr.server import Grid

from .strategy import Strategy


class DifferentialPrivacyServerSideFixedClipping(Strategy):
    """Strategy wrapper for central DP with server-side fixed clipping.

    Parameters
    ----------
    strategy : Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier : float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm : float
        The value of the clipping norm.
    num_sampled_clients : int
        The number of clients that are sampled on each round.

    Examples
    --------
    Create a strategy::

        strategy = fl.serverapp.FedAvg( ... )

    Wrap the strategy with the DifferentialPrivacyServerSideFixedClipping wrapper::

        dp_strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy, cfg.noise_multiplier, cfg.clipping_norm, cfg.num_sampled_clients
        )
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")

        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

        self.current_arrays: ArrayRecord = ArrayRecord()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "Differential Privacy Strategy Wrapper (Server-Side Fixed Clipping)"
        return rep

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of training."""
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: list[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        num_errors = 0
        for msg in replies:
            if msg.has_error():
                log(
                    INFO,
                    "Received error in reply from node %d: %s",
                    msg.metadata.src_node_id,
                    msg.error,
                )
                num_errors += 1

        # Errors are not allowed
        if num_errors:
            log(
                INFO,
                "aggregate_train: Some clients reported errors. Skipping aggregation.",
            )
            return None, None

        log(
            INFO,
            "aggregate_train: Received %s results and %s failures",
            len(replies) - num_errors,
            num_errors,
        )

        if len(replies) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(replies),
                self.num_sampled_clients,
            )

        # Clip arrays in replies
        current_ndarrays = self.current_arrays.to_numpy_ndarrays()
        for reply in replies:
            for arr_name, record in reply.content.array_records.items():
                # Clip
                reply_ndarrays = record.to_numpy_ndarrays()
                compute_clip_model_update(
                    param1=reply_ndarrays,
                    param2=current_ndarrays,
                    clipping_norm=self.clipping_norm,
                )
                # Replace
                reply.content[arr_name] = ArrayRecord(reply_ndarrays)
            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )

        # Pass the new parameters for aggregation
        aggregated_arrays, aggregated_metrics = self.strategy.aggregate_train(
            server_round, replies
        )

        # Add Gaussian noise to the aggregated arrays
        if aggregated_arrays:

            aggregated_ndarrays = aggregated_arrays.to_numpy_ndarrays()
            add_gaussian_noise_inplace(
                aggregated_ndarrays,
                compute_stdv(
                    self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
                ),
            )
            aggregated_arrays = ArrayRecord(aggregated_ndarrays)

            log(
                INFO,
                "aggregate_fit: central DP noise with %.4f stdev added",
                compute_stdv(
                    self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
                ),
            )

        return aggregated_arrays, aggregated_metrics
