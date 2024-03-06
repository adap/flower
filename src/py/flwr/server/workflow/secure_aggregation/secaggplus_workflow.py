# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Workflow for the SecAgg+ protocol."""


from __future__ import annotations

from flwr.server.driver import Driver
from flwr.common import Context, log, RecordSet, ConfigsRecord, Message
from flwr.server.compat.legacy_context import LegacyContext
import flwr.common.recordset_compat as compat
from flwr.common.constant import MESSAGE_TYPE_FIT
from flwr.common.secure_aggregation.secaggplus_constants import Key, Stage, RECORD_KEY_CONFIGS, RECORD_KEY_STATE
from dataclasses import dataclass, field
from logging import WARN, INFO, ERROR
from typing import List, Dict
from ..default_workflows import KEY_CURRENT_ROUND, CONFIGS_RECORD_KEY, PARAMS_RECORD_KEY
import random


LOG_EXPLAIN = True


@dataclass
class WorkflowState:
    """The state of the SecAgg+ protocol."""
    sampled_node_ids: set[int] = field(default_factory=set)
    active_node_ids: set[int] = field(default_factory=set)
    num_shares: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    quantization_range: int = 0
    mod_range: int = 0
    nid_to_neighbours: dict[int, set[int]] = {}


class SecAggPlusWorkflow:
    """The workflow for the SecAgg+ protocol.

    Parameters
    ----------
    num_shares : Union[int, float]
        The number of shares into which each client's private key is split under
        the SecAgg+ protocol. If specified as a float, it represents the proportion
        of all selected clients, and the number of shares will be set dynamically in
        the run time. A private key can be reconstructed from these shares, allowing
        for the secure aggregation of model updates. Each client sends one share to
        each of its neighbors while retaining one.
    reconstruction_threshold : Union[int, float]
        The minimum number of shares required to reconstruct a client's private key,
        or, if specified as a float, it represents the proportion of the total number
        of shares needed for reconstruction. This threshold ensures privacy by allowing
        for the recovery of contributions from dropped clients during aggregation,
        without compromising individual client data.
    max_weight : Optional[float] (default: 1000.0)
        The maximum value of the weight that can be assigned to any single client's 
        update during the weighted average calculation on the server side, e.g., in the 
        FedAvg algorithm.
    clipping_range : float, optional (default: 8.0)
        The range within which model parameters are clipped before quantization.
        This parameter ensures each model parameter is bounded within
        [-clipping_range, clipping_range], facilitating quantization.
    quantization_range : int, optional (default: 1048576, this equals 2**20)
        The size of the range into which floating-point model parameters are quantized,
        mapping each parameter to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 2147483648, this equals 2**30)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]).
    timeout : Optional[float] (default: None)
        The timeout duration in seconds. If specified, the workflow will wait for
        replies for this duration each time. If `None`, there is no time limit and 
        the workflow will wait until replies for all messages are received.

    Notes
    -----
    - Generally, higher `num_shares` means more robust to dropouts while increasing the
    computational costs; higher `reconstruction_threshold` means better privacy 
    guarantees but less tolerance to dropouts.
    - Too large `max_weight` may compromise the precision of the quantization.
    - `modulus_range` must be larger than `quantization_range`.
    - When `num_shares` is a float, it is interpreted as the proportion of all selected
    clients, and hence the number of shares will be determined in the runtime. This 
    allows for dynamic adjustment based on the total number of participating clients.
    - Similarly, when `reconstruction_threshold` is a float, it is interpreted as the
    proportion of the number of shares needed for the reconstruction of a private key.
    This feature enables flexibility in setting the security threshold relative to the
    number of distributed shares.
    - `num_shares`, `reconstruction_threshold`, and the quantization parameters
    (`clipping_range`, `target_quantization_range`, `modulus_range`) play critical roles
    in balancing privacy, robustness, and efficiency within the SecAgg+ protocol.
    """

    def __init__(
        self,
        num_shares: int | float,
        reconstruction_threshold: int | float,
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 1048576,
        modulus_range: int = 2147483648,
        timeout: float | None = None,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout
        
        self._check_init_params()
    
    def __call__(self, driver: Driver, context: Context) -> None:
        ...

    def _check_init_params(self) -> None:
        # Check `num_shares`
        if not isinstance(self.num_shares, (int, float)):
            raise TypeError("`num_shares` must be of type int or float.")
        if isinstance(self.num_shares, int):
            if self.num_shares <= 2:
                raise ValueError("`num_shares` as an integer must be greater than 2.")
            if self.num_shares > self.modulus_range / self.quantization_range:
                log(
                    WARN, 
                    "A `num_shares` larger than `mod_range / target_range` will potentially "
                    "cause overflow when computing the aggregated model parameters."
                )
            if self.num_shares == 1:
                self.num_shares = 1.0
        elif self.num_shares <= 0:
            raise ValueError("`num_shares` as a float must be greater than 0.")
        
        # Check `reconstruction_threshold`
        if not isinstance(self.reconstruction_threshold, (int, float)):
            raise TypeError("`reconstruction_threshold` must be of type int or float.")
        if isinstance(self.reconstruction_threshold, int):
            if isinstance(self.num_shares, int):
                if self.reconstruction_threshold >= self.num_shares:
                    raise ValueError(
                        "`reconstruction_threshold` must be less than `num_shares`."
                    )
            if self.reconstruction_threshold == 1:
                self.reconstruction_threshold = 1.0
        else:
            if not (0 < self.reconstruction_threshold <= 1):
                raise ValueError(
                    "If `reconstruction_threshold` is a float, "
                    "it must be greater than 0 and less than or equal to 1."
                )
        
        # Check `max_weight`
        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")
        
        # Check `quantization_range`
        if self.quantization_range <= 0:
            raise ValueError("`quantization_range` must be greater than 0.")
        
        # Check `target_range`
        if not isinstance(self.quantization_range, int) or self.quantization_range <= 0:
            raise ValueError("`target_range` must be an integer and greater than 0.")
        
        # Check `mod_range`
        if not isinstance(self.modulus_range, int) or self.modulus_range <= self.quantization_range:
            raise ValueError("`mod_range` must be an integer and greater than `target_range`.")

    def _check_threshold(self, state: WorkflowState) -> bool:
        for node_id in state.active_node_ids:
            active_neighbors = state.nid_to_neighbours[node_id]
            active_neighbors.intersection_update(state.active_node_ids)
            if len(active_neighbors) < state.threshold:
                log(ERROR, "Insufficient available nodes.")
                return False
        return True

    def _setup(self, driver: Driver, context: LegacyContext, state: WorkflowState) -> bool:
        context.strategy.configure_fit()
        
        # Protocol config
        state.sampled_node_ids = set(driver.get_node_ids())
        num_samples = len(state.sampled_node_ids)
        if num_samples < 2:
            log(ERROR, "The number of samples should be greater than 1.")
            return False
        if isinstance(self.num_shares, float):
            state.num_shares = round(self.num_shares * num_samples)
            # If even
            if state.num_shares < num_samples and state.num_shares & 1 == 0:
                state.num_shares += 1
            # If too small
            if state.num_shares <= 2:
                state.num_shares = num_samples
        if isinstance(self.reconstruction_threshold, float):
            state.threshold = round(self.reconstruction_threshold * state.num_shares)
            # If too small
            if state.threshold < 2:
                state.threshold = 2
        state.active_node_ids = state.sampled_node_ids
        state.clipping_range = self.clipping_range
        state.quantization_range = self.quantization_range
        state.mod_range = self.modulus_range
        
        sa_params_dict = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_samples,
            Key.SHARE_NUMBER: state.num_shares,
            Key.THRESHOLD: state.threshold,
            Key.CLIPPING_RANGE: state.clipping_range,
            Key.TARGET_RANGE: state.quantization_range,
            Key.MOD_RANGE: state.mod_range,
        }
        
        # The number of shares should better be odd in the SecAgg+ protocol.
        if num_samples != state.num_shares and state.num_shares & 1 == 0:
            log(WARN, "Number of shares in the SecAgg+ protocol should be odd.")
            state.num_shares += 1

        # Randomly assign secure IDs to clients
        random.shuffle(state.sampled_node_ids)
        # Build neighbour relations (node ID -> secure IDs of neighbours)
        half_share = state.num_shares >> 1
        state.nid_to_neighbours = {
            nid: {
                state.sampled_node_ids[(idx + offset) % num_samples]
                for offset in range(-half_share, half_share + 1)
            }
            for idx, nid in enumerate(state.sampled_node_ids)
        }

        state.active_node_ids = state.sampled_node_ids
        
        # Send setup configuration to clients
        cfgs = ConfigsRecord(sa_params_dict)
        content = RecordSet(configs_records={RECORD_KEY_CONFIGS: cfgs})
        def make(nid: int) -> Message:
            return driver.create_message(
                content=content,
                message_type=MESSAGE_TYPE_FIT,
                dst_node_id=nid,
                group_id="",
                ttl="",
            )
        
        msgs = driver.send_and_receive([
            make(node_id) for node_id in state.active_node_ids
        ], timeout=self.timeout)
        state.active_node_ids = {msg.metadata.src_node_id for msg in msgs if not msg.has_error()}
        if LOG_EXPLAIN:
            print(f"Received public keys from {len(state.active_node_ids)} clients.")

        nid2public_keys = {}
        for msg in msgs:
            if msg.has_error():
                continue
        return True


