import random
from dataclasses import dataclass
from logging import ERROR, INFO, WARN
from time import sleep
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


def aggregate(records: list[RecordDict], weighting_metric_name: str) -> RecordDict:
    # Aggregate the given records using the specified weighting metric
    aggregated = RecordDict()

    # Count total examples
    total_examples = 0
    examples_per_record = []
    for record in records:
        for rec in record.values():
            weight = None
            if isinstance(rec, MetricRecord):
                if weighting_metric_name in rec:
                    weight = rec[weighting_metric_name]
                    break  # move to next RecordDict

        if weight is None:
            # Warn user no weighting key is found and that 1.0 will be used
            log(
                WARN,
                "No weighting key '%s' found in MetricRecord, defaulting to 1.0",
                weighting_metric_name,
            )
            weight = 1.0
        total_examples += weight
        examples_per_record.append(weight)

    if total_examples == 0:
        # it could be that no `MetricRecord` was sent in the replies by ClientApps
        #! Warn users ?
        total_examples = len(records)
        examples_per_record = [1.0] * len(records)

    # Perform weighted aggregation
    for record, weight in zip(records, examples_per_record):
        for name, record_item in record.items():
            # For ArrayRecord
            if isinstance(record_item, ArrayRecord):
                if name not in aggregated:
                    aggregated[name] = ArrayRecord()
                # aggregate in-place
                # TODO: optimize so at least we don't need to serde all the time
                for key, value in record_item.items():
                    if key not in aggregated[name]:
                        aggregated[name][key] = Array(
                            value.numpy() * weight / total_examples
                        )
                    else:
                        aggregated[name][key] = Array(
                            aggregated[name][key].numpy()
                            + value.numpy() * weight / total_examples
                        )

            # For MetricRecord
            elif isinstance(record_item, MetricRecord):
                if name not in aggregated:
                    aggregated[name] = MetricRecord()
                # aggregate in-place
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        continue  # ! we don't want to keep this key as part of the aggregated recorddict, do we?
                    if key not in aggregated[name]:
                        if isinstance(value, list):
                            aggregated[name][key] = (
                                np.array(value) * weight / total_examples
                            ).tolist()
                        else:
                            aggregated[name][key] = value * weight / total_examples
                    else:
                        if isinstance(value, list):
                            aggregated[name][key] = (
                                np.array(aggregated[name][key])
                                + np.array(value) * weight / total_examples
                            ).tolist()
                        else:
                            aggregated[name][key] += value * weight / total_examples

    return aggregated


class FedAvg:

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_clients: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighting_factor_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ):
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_clients = min_train_clients
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

    def sample_nodes(self, grid: Grid, is_train: bool = False) -> list[int]:

        # Determinie minimum numbers to sample
        min_sample_nodes = (
            self.min_train_clients if is_train else self.min_evaluate_nodes
        )
        fraction = self.fraction_train if is_train else self.fraction_evaluate
        sampled_nodes = []
        while len(sampled_nodes) < min_sample_nodes:
            all_node_ids = list(grid.get_node_ids())
            to_sample = int(len(all_node_ids) * fraction)
            # Are there enough nodes that if sampled the specified fraction
            # the resulted sample size is equal or larger than the minimum required
            if to_sample >= min_sample_nodes:
                # Sample
                sampled_nodes = random.sample(all_node_ids, to_sample)
                break
            log(
                INFO,
                f"Waiting for nodes to connect. Nodes connected {len(all_node_ids)}.",
            )
            log(
                INFO,
                f"Configured to sample a minimum of {min_sample_nodes} nodes with sampling fraction set to {fraction} (over all connected nodes).",
            )
            sleep(3)

        log(
            INFO,
            f"Node sampling ({'train' if is_train else 'evaluate'}): sampled {len(sampled_nodes)} clients (out of {len(all_node_ids)})",
        )

        return sampled_nodes

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:

        # Sample nodes
        node_ids = self.sample_nodes(grid, is_train=True)

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
    ) -> RecordDict:
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

        rd = aggregate(
            records=[msg.content for msg in replies if msg.has_content()],
            weighting_metric_name="num-examples",
        )

        return rd

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        # Configure the next round of evaluation

        # Sample nodes
        node_ids = self.sample_nodes(grid, is_train=False)

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

        rd = aggregate(
            records=[msg.content for msg in replies if msg.has_content()],
            weighting_metric_name="num-examples",
        )
        return rd.metric_records

    def launch(
        self,
        arrays: ArrayRecord,  #! Now compulsory (those that don't what a mdoel, they can pass empyt ArrayRecord)
        grid: Grid,
        num_rounds: int,
        timeout: float,
        train_config: Optional[ConfigRecord] = ConfigRecord(),
        evaluate_config: Optional[ConfigRecord] = ConfigRecord(),
        central_eval_fn: Callable[[int, RecordDict], MetricRecord] = None,
    ) -> MetricRecord:

        # Log brief info about Strategy setup

        # log name of strategy
        log(INFO, f"Starting {self.__class__.__name__} strategy.")
        log(INFO, f"\tnum_rounds: {num_rounds}")
        log(
            INFO,
            f"\tarray_record: {len(arrays)} Arrays totalling {sum(np.prod(array.shape) for array in arrays.values())/(1000**2):.2f} M parameters",
        )
        log(
            INFO,
            f"\tConfig for train round: {train_config if train_config else '⚠️ (empty!)'}",
        )
        log(
            INFO,
            f"\tConfig for evaluate round: {evaluate_config if evaluate_config else '⚠️ (empty!)'}",
        )
        log(INFO, "")

        metrics_history = ReturnStrategyResults()

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
            # Communicate
            replies = grid.send_and_receive(messages, timeout=timeout)
            # aggregate fit
            res: RecordDict = self.aggregate_train(current_round, replies)
            # copy the content of the array records in res to those in arrays
            # ! Change: make aggregate_train return tuple[arrayrecord, metricrecord]
            for key, record in res.array_records.items():
                arrays = record
            # Log training metrics and append to history
            log(INFO, "Federated Training results: %s", res.metric_records)
            metrics_history.train_metrics[current_round] = res.metric_records

            # Configure evaluate
            messages = self.configure_evaluate(
                current_round, arrays, evaluate_config, grid
            )
            # Communicate
            replies = grid.send_and_receive(messages, timeout=timeout)
            # Aggregate evaluate
            eval_res = self.aggregate_evaluate(current_round, replies)
            log(INFO, "Federated Evaluation results: %s", eval_res)
            metrics_history.evaluate_metrics[current_round] = eval_res

            # Centralised eval
            if central_eval_fn:
                res = central_eval_fn(server_round=current_round, array_record=arrays)
                log(INFO, "Central evaluation results: %s", res)
                metrics_history.central_evaluate_metrics[current_round] = res

        log(INFO, "Finished all rounds")
        log(INFO, "")

        return metrics_history
