import random
from dataclasses import dataclass, field
from logging import ERROR, INFO
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
    train_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    central_evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    arrays: ArrayRecord = None


def aggregate_arrayrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> ArrayRecord:
    """Perform weighted aggregation all ArrayRecords using a specific key."""

    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        weights.append(metricrecord[weighting_metric_name])

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    # Perform weighted aggregation
    aggregated_np_arrays: dict[str, np.NDArray] = {}

    for record, weight in zip(records, weight_factors):
        for record_item in record.values():
            # For ArrayRecord
            if isinstance(record_item, ArrayRecord):
                # aggregate in-place
                for key, value in record_item.items():
                    if key not in aggregated_np_arrays:
                        aggregated_np_arrays[key] = value.numpy() * weight
                    else:
                        aggregated_np_arrays[key] += value.numpy() * weight

    return ArrayRecord({k: Array(v) for k, v in aggregated_np_arrays.items()})


def aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""

    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        weights.append(metricrecord[weighting_metric_name])

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors):
        for record_item in record.values():
            # For MetricRecord
            if isinstance(record_item, MetricRecord):
                # aggregate in-place
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        # We exclude the weighting key from the aggregated MetricRecord
                        continue
                    if key not in aggregated_metrics:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(value) * weight
                            ).tolist()
                        else:
                            aggregated_metrics[key] = value * weight
                    else:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(aggregated_metrics[key])
                                + np.array(value) * weight
                            ).tolist()
                        else:
                            aggregated_metrics[key] += value * weight

    return aggregated_metrics


def sample_nodes(
    grid: Grid, min_available_nodes: int, sample_size: int
) -> tuple[list[int], list[int]]:
    """Sample the specified number of nodes using the Grid."""

    sampled_nodes = []

    # wait for min_available_nodes to be online
    nodes_connected = grid.get_node_ids()
    while len(nodes_connected) < min_available_nodes:
        sleep(1)
        log(
            INFO,
            f"Waiting for nodes to connect. Nodes connected {len(nodes_connected)} "
            f"(expecting at least {min_available_nodes}).",
        )
        nodes_connected = grid.get_node_ids()

    # Sample nodes
    sampled_nodes = random.sample(list(nodes_connected), sample_size)

    return sampled_nodes, nodes_connected


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
        train_metrics_aggregation_fn: Callable[
            [list[RecordDict], str], MetricRecord
        ] = aggregate_metricrecords,
        evaluate_metrics_aggregation_fn: Callable[
            [list[RecordDict], str], MetricRecord
        ] = aggregate_metricrecords,
    ):
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighting_factor_key = weighting_factor_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.train_metrics_aggregation_fn = train_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: MessageType
    ) -> list[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )  #! Note i'm not using group_id
            messages.append(message)
        return messages

    def _check_message_reply_consistency(
        self, replies: list[RecordDict], check_arrayrecord: bool
    ) -> bool:
        """Check that replies contain one ArrayRecord, OneMetricRecord and that
        the weighting factor key is present."""

        # Checking for ArrayRecord consistency
        skip_aggregation = False
        if check_arrayrecord:
            if all(len(msg.array_records) != 1 for msg in replies):
                log(
                    ERROR,
                    "Expected exactly one ArrayRecord in replies, but found more. "
                    "Skipping aggregation.",
                )
                skip_aggregation = True
            else:
                # Ensure all key are present in all ArrayRecords
                all_key_sets = [
                    set(next(iter(d.array_records.values())).keys()) for d in replies
                ]
                if not all(s == all_key_sets[0] for s in all_key_sets):
                    log(
                        ERROR,
                        "All ArrayRecords must have the same keys for aggregation. "
                        "This condition wasn't met. Skipping aggregation.",
                    )
                    skip_aggregation = True

        # Checking for MetricRecord consistency
        if all(len(msg.metric_records) != 1 for msg in replies):
            log(
                ERROR,
                "Expected exactly one MetricRecord in replies, but found more. "
                "Skipping aggregation.",
            )
            skip_aggregation = True
        else:
            # Ensure all key are present in all MetricRecords
            all_key_sets = [
                set(next(iter(d.metric_records.values())).keys()) for d in replies
            ]
            if not all(s == all_key_sets[0] for s in all_key_sets):
                log(
                    ERROR,
                    "All MetricRecords must have the same keys for aggregation. "
                    "This condition wasn't met. Skipping aggregation.",
                )
                skip_aggregation = True

            # Check one of the sets for the key to perform weighting averaging
            if self.weighting_factor_key not in all_key_sets[0]:
                log(
                    ERROR,
                    f"The MetricRecord in the reply messages were expecting key `{self.weighting_factor_key}` "
                    "to perform averaging of ArrayRecords and MetricRecords. Skipping aggregation.",
                )
                skip_aggregation = True

        return skip_aggregation

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of federated training."""

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
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""

        # Log if any Messages carried errors
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

        # Filter messages that carry content
        replies_with_content = [msg.content for msg in replies if msg.has_content()]

        # Ensure expected ArrayRecords and MetricRecords are received
        skip_aggregation = self._check_message_reply_consistency(
            replies=replies_with_content, check_arrayrecord=True
        )

        if skip_aggregation:
            return None, None

        # Aggregate ArrayRecords
        arrays = aggregate_arrayrecords(
            records=replies_with_content,
            weighting_metric_name=self.weighting_factor_key,
        )

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggregation_fn(
            records=replies_with_content,
            weighting_metric_name=self.weighting_factor_key,
        )
        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of federated evaluation."""

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
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""

        # Log if any Messages carried errors
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

        # Filter messages that carry content
        replies_with_content = [msg.content for msg in replies if msg.has_content()]

        # Ensure expected ArrayRecords and MetricRecords are received
        skip_aggregation = self._check_message_reply_consistency(
            replies=replies_with_content, check_arrayrecord=False
        )

        if skip_aggregation:
            return None

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggregation_fn(
            records=replies_with_content,
            weighting_metric_name=self.weighting_factor_key,
        )
        return metrics

    def start(
        self,
        arrays: ArrayRecord,
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
        log(INFO, f"\t└──> Number of rounds: {num_rounds}")
        log(
            INFO,
            f"\t└──> ArrayRecord: {len(arrays)} Arrays totalling {sum(len(array.data) for array in arrays.values())/(1024**2):.2f} MB",
        )
        log(
            INFO,
            f"\t└──> ConfigRecord (train): {train_config if train_config else '(empty!)'}",
        )
        log(
            INFO,
            f"\t└──> ConfigRecord (evaluate): {evaluate_config if evaluate_config else '(empty!)'}",
        )
        log(INFO, "")

        metrics_history = ReturnStrategyResults()

        t_start = time()
        # Do central eval with starting global parameters
        if central_eval_fn:
            res = central_eval_fn(server_round=0, array_record=arrays)
            log(INFO, "Initial central evaluation results: %s", res)
            metrics_history.central_evaluate_metrics[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Configure train, send messages and wait for replies
            replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round, arrays, train_config, grid
                ),
                timeout=timeout,
            )
            # Aggregate train
            arrays, agg_metrics = self.aggregate_train(current_round, replies)
            # Log training metrics and append to history
            # TODO: this is the most strinct mode of opeartion (if aggregation is skipped due to inconsistent replies)
            if arrays is None or agg_metrics is None:
                break
            log(INFO, "\t└──> Aggregated  MetricRecord: %s", agg_metrics)
            metrics_history.train_metrics[current_round] = agg_metrics
            metrics_history.arrays = arrays

            # Configure evaluate, send messages and wait for replies
            replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round, arrays, evaluate_config, grid
                ),
                timeout=timeout,
            )
            # Aggregate evaluate
            agg_metrics = self.aggregate_evaluate(current_round, replies)
            # TODO: this is the most strinct mode of opeartion (if aggregation is skipped due to inconsistent replies)
            if arrays is None or agg_metrics is None:
                break
            log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_metrics)
            metrics_history.evaluate_metrics[current_round] = agg_metrics

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
