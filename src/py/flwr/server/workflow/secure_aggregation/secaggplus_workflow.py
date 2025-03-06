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


from collections.abc import Iterable
from logging import INFO
from typing import Optional, Union, cast

import flwr.common.recordset_compat as compat
from flwr.common import (
    Context,
    FitRes,
    Message,
    MessageType,
    RecordSet,
    log,
    ndarrays_to_parameters,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import factor_extract
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    Key,
    Stage,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.driver import Driver

from ..constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from ..constant import Key as WorkflowKey
from .secaggplus_aggregator import SecAggPlusAggregator, SecAggPlusAggregatorState


class SecAggPlusWorkflow:  # pylint: disable=too-many-instance-attributes
    """The workflow for the SecAgg+ protocol.

    Please use with the `secaggplus_mod` modifier on the ClientApp side.

    The SecAgg+ protocol ensures the secure summation of integer vectors owned by
    multiple parties, without accessing any individual integer vector. This workflow
    allows the server to compute the weighted average of model parameters across all
    clients, ensuring individual contributions remain private. This is achieved by
    clients sending both, a weighting factor and a weighted version of the locally
    updated parameters, both of which are masked for privacy. Specifically, each
    client uploads "[w, w * params]" with masks, where weighting factor 'w' is the
    number of examples ('num_examples') and 'params' represents the model parameters
    ('parameters') from the client's `FitRes`. The server then aggregates these
    contributions to compute the weighted average of model parameters.

    The protocol involves four main stages:

    - 'setup': Send SecAgg+ configuration to clients and collect their public keys.
    - 'share keys': Broadcast public keys among clients and collect encrypted secret
      key shares.
    - 'collect masked vectors': Forward encrypted secret key shares to target clients
      and collect masked model parameters.
    - 'unmask': Collect secret key shares to decrypt and aggregate the model parameters.

    Only the aggregated model parameters are exposed and passed to
    `Strategy.aggregate_fit`, ensuring individual data privacy.

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
    quantization_range : int, optional (default: 4194304, this equals 2**22)
        The size of the range into which floating-point model parameters are quantized,
        mapping each parameter to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 4294967296, this equals 2**32)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]). `modulus_range` must be less than 4294967296.
        Please use 2**n values for `modulus_range` to prevent overflow issues.
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
    - `modulus_range` must be 2**n and larger than `quantization_range`.
    - When `num_shares` is a float, it is interpreted as the proportion of all selected
      clients, and hence the number of shares will be determined in the runtime. This
      allows for dynamic adjustment based on the total number of participating clients.
    - Similarly, when `reconstruction_threshold` is a float, it is interpreted as the
      proportion of the number of shares needed for the reconstruction of a private key.
      This feature enables flexibility in setting the security threshold relative to the
      number of distributed shares.
    - `num_shares`, `reconstruction_threshold`, and the quantization parameters
      (`clipping_range`, `quantization_range`, `modulus_range`) play critical roles in
      balancing privacy, robustness, and efficiency within the SecAgg+ protocol.
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_shares: Union[int, float],
        reconstruction_threshold: Union[int, float],
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: Optional[float] = None,
    ) -> None:
        # Check `max_weight`
        if max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout

        # Runtime attributes
        self._current_round: int = -1
        self._legacy_results: list[tuple[ClientProxy, FitRes]] = []
        self._legacy_failures: list[Exception] = []
        self._nid_to_fitins: dict[int, RecordSet] = {}
        self._nid_to_proxies: dict[int, ClientProxy] = {}
        self._legacy_context: Optional[LegacyContext] = None

    def __call__(self, driver: Driver, context: Context) -> None:
        """Run the SecAgg+ protocol."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )
        self._legacy_context = context

        # Obtain fit instructions and initialize
        cfg = context.state.configs_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])
        parameters = compat.parametersrecord_to_parameters(
            context.state.parameters_records[MAIN_PARAMS_RECORD],
            keep_input=True,
        )
        proxy_fitins_lst = context.strategy.configure_fit(
            current_round, parameters, context.client_manager
        )
        if not proxy_fitins_lst:
            log(INFO, "configure_fit: no clients selected, cancel")
            return
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(proxy_fitins_lst),
            context.client_manager.num_available(),
        )
        self._current_round = current_round
        self._nid_to_fitins = {
            proxy.node_id: compat.fitins_to_recordset(fitins, True)
            for proxy, fitins in proxy_fitins_lst
        }
        self._nid_to_proxies = {proxy.node_id: proxy for proxy, _ in proxy_fitins_lst}

        # Create the aggregator
        aggregator = SecAggPlusAggregator(
            driver=driver,
            context=context,
            num_shares=self.num_shares,
            reconstruction_threshold=self.reconstruction_threshold,
            clipping_range=self.clipping_range,
            quantization_range=self.quantization_range,
            modulus_range=self.modulus_range,
            timeout=self.timeout,
            on_send=self.on_send,
            on_receive=self.on_receive,
            on_stage_complete=self.on_stage_complete,
        )

        aggregator.aggregate(
            [
                driver.create_message(
                    content=content,
                    message_type=MessageType.TRAIN,
                    dst_node_id=nid,
                    group_id=str(current_round),
                )
                for nid, content in self._nid_to_fitins.items()
            ]
        )

        # Clean up
        self._legacy_results.clear()
        self._legacy_failures.clear()
        self._nid_to_fitins.clear()
        self._nid_to_proxies.clear()

    def on_send(
        self, msgs: Iterable[Message], state: SecAggPlusAggregatorState
    ) -> None:
        """Handle messages to be sent."""
        if state.current_stage == Stage.COLLECT_MASKED_VECTORS:
            for msg in msgs:
                cfg = msg.content.configs_records[RECORD_KEY_CONFIGS]
                cfg[Key.MAX_WEIGHT] = self.max_weight
                cfg[Key.CLIPPING_RANGE] = self.clipping_range

    def on_receive(
        self, msgs: Iterable[Message], state: SecAggPlusAggregatorState
    ) -> None:
        """Handle received reply messages."""
        for msg in msgs:
            if msg.has_error():
                self._legacy_failures.append(Exception(msg.error))
        if state.current_stage == Stage.COLLECT_MASKED_VECTORS:
            for msg in msgs:
                fitres = compat.recordset_to_fitres(msg.content, True)
                proxy = self._nid_to_proxies[msg.metadata.src_node_id]
                self._legacy_results.append((proxy, fitres))

    def on_stage_complete(
        self, success: bool, state: SecAggPlusAggregatorState
    ) -> None:
        """Handle stage completion."""
        if not success or state.current_stage != Stage.UNMASK:
            return
        if self._legacy_context is None:
            raise RuntimeError(
                "Unexpected error: LegacyContext is not set. This typically indicates "
                "that this instance is corrupted or in an invalid state."
            )
        context = self._legacy_context

        # Post-process after a successful unmask stage
        total_scaled_ratio, aggregated_vector = factor_extract(state.aggregated_vector)
        inv_total_ratio = state.clipping_range / total_scaled_ratio
        for vec in aggregated_vector:
            vec *= inv_total_ratio

        # Backward compatibility with Strategy
        parameters = ndarrays_to_parameters(aggregated_vector)
        for _, fitres in self._legacy_results:
            fitres.parameters = parameters

        # No exception/failure handling currently
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(self._legacy_results),
            len(self._legacy_failures),
        )
        aggregated_result = context.strategy.aggregate_fit(
            self._current_round,
            self._legacy_results,
            self._legacy_failures,  # type: ignore
        )
        parameters_aggregated, metrics_aggregated = aggregated_result

        # Update the parameters and write history
        if parameters_aggregated:
            paramsrecord = compat.parameters_to_parametersrecord(
                parameters_aggregated, True
            )
            context.state.parameters_records[MAIN_PARAMS_RECORD] = paramsrecord
            context.history.add_metrics_distributed_fit(
                server_round=self._current_round, metrics=metrics_aggregated
            )
