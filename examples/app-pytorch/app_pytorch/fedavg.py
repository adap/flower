import random
from dataclasses import dataclass
from logging import INFO, WARN
from time import sleep, time
from typing import Callable, Optional

import numpy as np
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid


@dataclass
class ReturnStrategyResults:
    train_metrics: dict[int, MetricRecord] = None
    evaluate_metrics: dict[int, MetricRecord] = None
    central_evaluate_metrics: dict[int, MetricRecord] = None
    arrays: ArrayRecord = None

    def __post_init__(self):
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.evaluate_metrics is None:
            self.evaluate_metrics = {}
        if self.central_evaluate_metrics is None:
            self.central_evaluate_metrics = {}
        if self.arrays is None:
            self.arrays = ArrayRecord()


def aggregate(
    records: list[RecordDict], weighting_metric_name: str
) -> tuple[ArrayRecord, MetricRecord]:
    # Aggregate the given records using the specified weighting metric

    # Count total examples
    total_examples = 0
    examples_per_record = []
    for record in records:
        weight = None
        # Check if a MetricRecord under key 'metrics' exist
        if mrecord := record.metric_records.get("metrics", None):
            # Check if a metric under key weighting_metric_name exists
            weight = mrecord.get(weighting_metric_name, None)
        if weight is None:
            log(
                WARN,
                "No weighting key '%s' found in MetricRecord, defaulting to 1.0 for all Messages",
                weighting_metric_name,
            )
            log(
                WARN,
                "Ensure your ClientApps include in their replies a {'metrics': MetricRecord({'%s': ...})}",
                weighting_metric_name,
            )
            total_examples = len(records)
            examples_per_record = [1.0] * len(records)
            break
        else:
            total_examples += weight
            examples_per_record.append(weight)

    # Perform weighted aggregation
    aggregated_arrays = ArrayRecord()
    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, examples_per_record):
        for record_item in record.values():
            # For ArrayRecord
            if isinstance(record_item, ArrayRecord):
                # aggregate in-place
                # TODO: optimize so at least we don't need to serde all the time
                for key, value in record_item.items():
                    if key not in aggregated_arrays:
                        aggregated_arrays[key] = Array(
                            value.numpy() * weight / total_examples
                        )
                    else:
                        aggregated_arrays[key] = Array(
                            aggregated_arrays[key].numpy()
                            + value.numpy() * weight / total_examples
                        )

            # For MetricRecord
            elif isinstance(record_item, MetricRecord):
                # aggregate in-place
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        continue  # ! we don't want to keep this key as part of the aggregated recorddict, do we?
                    if key not in aggregated_metrics:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(value) * weight / total_examples
                            ).tolist()
                        else:
                            aggregated_metrics[key] = value * weight / total_examples
                    else:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(aggregated_metrics[key])
                                + np.array(value) * weight / total_examples
                            ).tolist()
                        else:
                            aggregated_metrics[key] += value * weight / total_examples

    return aggregated_arrays, aggregated_metrics


def sample_nodes(
    grid: Grid, min_available_nodes: int, sample_size: int
) -> tuple[list[int], list[int]]:

    sampled_nodes = []

    # wait for min_available_nodes to be online
    nodes_connected = grid.get_node_ids()
    while len(nodes_connected) < min_available_nodes:
        sleep(1)
        log(
            INFO,
            f"Waiting for nodes to connect. Nodes connected {len(nodes_connected)} (expecting at least {min_available_nodes}).",
        )
        nodes_connected = grid.get_node_ids()

    # Sample nodes
    sampled_nodes = random.sample(list(nodes_connected), sample_size)

    return sampled_nodes, nodes_connected


def wait_for_replies(grid: Grid, msg_ids: list[str], timeout: float) -> list[Message]:

    unique_msg_ids = set(msg_ids)  # Ensure unique message IDs
    # Pull messages
    end_time = time() + (timeout if timeout is not None else 0.0)
    replies: list[Message] = []
    while timeout is None or time() < end_time:
        res_msgs = grid.pull_messages(unique_msg_ids)
        replies.extend(res_msgs)
        unique_msg_ids.difference_update(
            {msg.metadata.reply_to_message_id for msg in res_msgs}
        )
        if len(unique_msg_ids) == 0:
            break

        # TODO: update with grid.pull_interval
        sleep(0.1)

    return replies


class FedAvg:

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighting_factor_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ):
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighting_factor_key = weighting_factor_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: MessageType
    ) -> list[Message]:
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )  #! Note i'm not using group_id
            messages.append(message)
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:

        # Sample nodes
        num_nodes = int(len(grid.get_node_ids()) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: list[Message],
    ) -> tuple[ArrayRecord, MetricRecord]:
        # Aggregate training results

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

        log(
            INFO,
            "aggregate_train: received %s results and %s failures",
            len(replies) - num_errors,
            num_errors,
        )

        replies_with_content = [msg.content for msg in replies if msg.has_content()]
        # Check if replies have one `ArrayRecord` and at most one `MetricRecord`
        if all(len(msg.array_records) != 1 for msg in replies_with_content):
            log(
                WARN,
                "Expected exactly one ArrayRecord in replies, but found: %s",
                [len(msg.array_records) for msg in replies_with_content],
            )
            #! Should return and break for round loop
        if all(len(msg.metric_records) > 1 for msg in replies_with_content):
            log(
                WARN,
                "Expected at most one MetricRecord in replies, but found: %s",
                [len(msg.metric_records) for msg in replies_with_content],
            )
            #! Should return and break for round loop

        arrays, metrics = aggregate(
            records=replies_with_content,
            weighting_metric_name=self.weighting_factor_key,
        )

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        # Configure the next round of evaluation

        # Sample nodes
        num_nodes = int(len(grid.get_node_ids()) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: list[Message],
    ) -> MetricRecord:
        # Aggregate evaluation results

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

        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(replies) - num_errors,
            num_errors,
        )

        replies_with_content = [msg.content for msg in replies if msg.has_content()]
        # Check if replies have at most one `MetricRecord`
        if all(len(msg.metric_records) > 1 for msg in replies_with_content):
            log(
                WARN,
                "Expected at most one MetricRecord in replies, but found: %s",
                [len(msg.metric_records) for msg in replies_with_content],
            )
            #! Should return and break for round loop

        _, metrics = aggregate(
            records=replies_with_content,
            weighting_metric_name=self.weighting_factor_key,
        )
        return metrics

    def launch(
        self,
        arrays: ArrayRecord,  #! Now compulsory (those that don't what a mdoel, they can pass empyt ArrayRecord)
        grid: Grid,
        num_rounds: int,
        timeout: float,
        train_config: Optional[ConfigRecord] = ConfigRecord(),
        evaluate_config: Optional[ConfigRecord] = ConfigRecord(),
        central_eval_fn: Callable[[int, RecordDict], MetricRecord] = None,
    ) -> ReturnStrategyResults:

        # Log brief info about Strategy setup

        # log name of strategy
        log(INFO, f"Starting {self.__class__.__name__} strategy.")
        log(INFO, f"\t> Number of rounds: {num_rounds}")
        log(
            INFO,
            f"\t> ArrayRecord: {len(arrays)} Arrays totalling {sum(len(array.data) for array in arrays.values())/(1024**2):.2f} MB",
        )
        log(
            INFO,
            f"\t> ConfigRecord for train round: {train_config if train_config else '(empty!)'}",
        )
        log(
            INFO,
            f"\t> ConfigRecord for evaluate round: {evaluate_config if evaluate_config else '(empty!)'}",
        )
        log(INFO, "")

        metrics_history = ReturnStrategyResults()

        t_start = time()
        # do central eval with starting global parameters
        if central_eval_fn:
            res = central_eval_fn(server_round=0, array_record=arrays)
            log(INFO, "Central evaluation results: %s", res)
            metrics_history.central_evaluate_metrics[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Configure train
            messages = self.configure_train(current_round, arrays, train_config, grid)
            # Send messages
            msg_ids = grid.push_messages(messages)
            del messages
            # Wait until replies are received
            replies = wait_for_replies(grid, msg_ids, timeout=timeout)
            # Aggregate train
            arrays, agg_metrics = self.aggregate_train(current_round, replies)
            # Log training metrics and append to history
            log(INFO, "\t└──> Aggregated  MetricRecord: %s", agg_metrics)
            metrics_history.train_metrics[current_round] = agg_metrics
            metrics_history.arrays = arrays

            # Configure evaluate
            messages = self.configure_evaluate(
                current_round, arrays, evaluate_config, grid
            )
            # Send messages
            msg_ids = grid.push_messages(messages)
            del messages
            # Wait until replies are received
            replies = wait_for_replies(grid, msg_ids, timeout=timeout)
            # Aggregate evaluate
            eval_res = self.aggregate_evaluate(current_round, replies)
            log(INFO, "\t└──> Aggregated MetricRecord: %s", eval_res)
            metrics_history.evaluate_metrics[current_round] = eval_res

            # Centralised eval
            if central_eval_fn:
                log(INFO, "Central evaluation")
                res = central_eval_fn(server_round=current_round, array_record=arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                metrics_history.central_evaluate_metrics[current_round] = res

        log(INFO, "")
        log(INFO, f"Strategy execution finished in {time() - t_start:.2f}s")
        log(INFO, "")

        return metrics_history
