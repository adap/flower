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
"""Flower message-based FedAvg strategy with layer-wise aggregation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import io
from logging import INFO, WARNING
from time import perf_counter
import time
from typing import Any

import torch

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    GRPC_MAX_MESSAGE_LENGTH,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.common.profiling import get_active_profiler, set_current_round
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    log_strategy_start_info,
    sample_nodes,
)

SAFE_GRPC_BYTES = int(GRPC_MAX_MESSAGE_LENGTH * 0.75)


def _chunk_slices(tensor: torch.Tensor, max_bytes: int) -> list[tuple[int, int]]:
    """Return (start, end) slices along dim0 to fit under max_bytes."""
    if tensor.ndim == 0:
        return [(0, 1)]
    total_bytes = tensor.numel() * tensor.element_size()
    if total_bytes <= max_bytes:
        return [(0, tensor.shape[0])]
    bytes_per_row = tensor[0:1].numel() * tensor.element_size()
    rows_per_chunk = max(1, int(max_bytes // max(bytes_per_row, 1)))
    slices: list[tuple[int, int]] = []
    start = 0
    while start < tensor.shape[0]:
        end = min(start + rows_per_chunk, tensor.shape[0])
        slices.append((start, end))
        start = end
    return slices


def _has_meta_tensors(state_dict: dict[str, Any]) -> bool:
    for value in state_dict.values():
        if hasattr(value, "is_meta") and value.is_meta:
            return True
    return False


# pylint: disable=too-many-instance-attributes
class FedAvgStreaming(FedAvg):
    """Federated Averaging strategy with layer-wise aggregation."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        initial_state_dict: dict[str, Any] | None = None,
        max_chunk_bytes: int = SAFE_GRPC_BYTES,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )
        self._state_dict = initial_state_dict
        self._layer_names: list[str] | None = None
        self._max_chunk_bytes = max_chunk_bytes

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training (no arrays sent)."""
        if self.fraction_train == 0.0:
            return []
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        config["server-round"] = server_round
        if self._layer_names is not None:
            config["layer_names"] = self._layer_names
            config["num_layers"] = len(self._layer_names)

        record = RecordDict({self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate MetricRecords only (arrays are streamed separately)."""
        valid_replies, _ = self._check_and_log_replies(
            replies, is_train=True, validate=False
        )

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return None, metrics

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy with layer-wise aggregation."""
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        if self._state_dict is None:
            if len(initial_arrays) == 0:
                raise ValueError("initial_state_dict required when initial_arrays empty")
            self._state_dict = initial_arrays.to_torch_state_dict()
        state_dict = self._state_dict
        self._layer_names = list(state_dict.keys())

        t_start = time.time()
        if evaluate_fn:
            if _has_meta_tensors(state_dict):
                log(
                    WARNING,
                    "Skipping initial evaluation: model contains meta tensors.",
                )
            else:
                res = evaluate_fn(0, ArrayRecord(state_dict))
                log(INFO, "Initial global evaluation results: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            set_current_round(current_round)
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    initial_arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate metrics only
            _, agg_train_metrics = self.aggregate_train(current_round, train_replies)
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            node_ids = [
                msg.metadata.src_node_id for msg in train_replies if not msg.has_error()
            ]
            if not node_ids:
                log(WARNING, "No valid training replies, skipping round %s", current_round)
                continue

            # -----------------------------------------------------------------
            # --- LAYER-WISE COMMUNICATION -----------------------------------
            # -----------------------------------------------------------------
            for layer_idx, layer_name in enumerate(self._layer_names):
                tensor = state_dict[layer_name]
                cpu_tensor = tensor.detach().cpu() if hasattr(tensor, "cpu") else tensor
                chunk_slices = _chunk_slices(cpu_tensor, self._max_chunk_bytes)
                agg_tensor = torch.empty_like(cpu_tensor)

                for chunk_idx, (start, end) in enumerate(chunk_slices):
                    config = ConfigRecord(
                        {
                            "layer_idx": layer_idx,
                            "chunk_idx": chunk_idx,
                            "chunk_start": start,
                            "chunk_end": end,
                            "chunk_count": len(chunk_slices),
                        }
                    )
                    record = RecordDict({self.configrecord_key: config})
                    replies = grid.send_and_receive(
                        messages=self._construct_messages(
                            record,
                            node_ids,
                            message_type="train.layer_wise_communication",
                        ),
                        timeout=timeout,
                    )
                    valid_replies, _ = self._check_and_log_replies(
                        replies, is_train=True
                    )
                    if not valid_replies:
                        continue

                    reply_contents = [msg.content for msg in valid_replies]
                    profiler = get_active_profiler()
                    start_time = perf_counter() if profiler is not None else None
                    agg_record = aggregate_arrayrecords(
                        reply_contents,
                        self.weighted_by_key,
                    )
                    if profiler is not None and start_time is not None:
                        duration_ms = (perf_counter() - start_time) * 1000.0
                        profiler.record(
                            scope="server",
                            task="aggregate",
                            round=current_round,
                            node_id=None,
                            duration_ms=duration_ms,
                            metadata={"layer_idx": layer_idx, "chunk_idx": chunk_idx},
                        )

                    chunk_np = agg_record[layer_name].numpy()
                    chunk_tensor = torch.from_numpy(chunk_np)

                    if tensor.ndim == 0:
                        agg_tensor = chunk_tensor
                    else:
                        agg_tensor[start:end] = chunk_tensor

                state_dict[layer_name] = agg_tensor

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------
            if self.fraction_evaluate > 0.0:
                arrays = ArrayRecord(state_dict)
                evaluate_replies = grid.send_and_receive(
                    messages=self.configure_evaluate(
                        current_round,
                        arrays,
                        evaluate_config,
                        grid,
                    ),
                    timeout=timeout,
                )
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round,
                    evaluate_replies,
                )
                if agg_evaluate_metrics is not None:
                    log(
                        INFO,
                        "\t└──> Aggregated MetricRecord: %s",
                        agg_evaluate_metrics,
                    )
                    result.evaluate_metrics_clientapp[current_round] = (
                        agg_evaluate_metrics
                    )

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------
            if evaluate_fn:
                if _has_meta_tensors(state_dict):
                    log(
                        WARNING,
                        "Skipping global evaluation: model contains meta tensors.",
                    )
                else:
                    arrays = ArrayRecord(state_dict)
                    log(INFO, "Global evaluation")
                    res = evaluate_fn(current_round, arrays)
                    log(INFO, "\t└──> MetricRecord: %s", res)
                    if res is not None:
                        result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        if _has_meta_tensors(state_dict):
            log(
                WARNING,
                "Skipping final ArrayRecord: model contains meta tensors.",
            )
        else:
            final_arrays = ArrayRecord(state_dict)
            result.arrays = final_arrays
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
