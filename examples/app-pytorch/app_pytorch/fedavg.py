import random
from logging import INFO, WARN
from time import sleep

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


def aggregate(records: list[RecordDict], weighting_metric_name: str) -> RecordDict:
    # Aggregate the given records using the specified weighting metric
    aggregated = RecordDict()

    # Count total examples
    total_examples = 0
    examples_per_record = []
    for record in records:
        for rec in record.values():
            if isinstance(rec, MetricRecord):
                if weighting_metric_name not in rec:
                    #! alert user expected key is not found ?
                    # default to homogeneous weighting (i.e. 1.0
                    # for all model contributions)
                    log(WARN, "not weighting key found")
                weight = rec.get(weighting_metric_name, 1)
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
        weighting_key: str = "num-examples",
        clientapp_train_config_key: str = "clientapp-train-config",
        clientapp_evaluate_config_key: str = "clientapp-evaluate-config",
    ):
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_clients = min_train_clients
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighting_key = weighting_key
        self.clientapp_train_config_key = clientapp_train_config_key
        self.clientapp_evaluate_config_key = clientapp_evaluate_config_key

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
        self, server_round: int, record: RecordDict, node_ids: list[int]
    ) -> list[Message]:
        # Configure the next round of training

        if self.clientapp_train_config_key not in record:
            # warn once the user that no configRecord was found
            # with that specific key. Therefore a new ConfigRecord is going
            # to be created. If it was specified, a new entry to the configrecord
            # would be added simply injecting the server_round.
            log(
                WARN,
                f"no ConfigRecord provided with `{self.clientapp_train_config_key}` key ....",
            )
            record[self.clientapp_train_config_key] = ConfigRecord(
                {"server_round": server_round}
            )
        else:
            # The user provided a `ConfigRecord` to send to `ClientApps`
            # when doing TRAIN. Here we append the round number
            record[self.clientapp_train_config_key]["server-round"] = server_round

        # Construct messages
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
        self, server_round: int, record: RecordDict, node_ids: list[int]
    ) -> list[Message]:
        # Configure the next round of evaluation

        if self.clientapp_evaluate_config_key not in record:
            # warn once the user that no configRecord was found
            # with that specific key. Therefore a new ConfigRecord is going
            # to be created. If it was specified, a new entry to the configrecord
            # would be added simply injecting the server_round.
            log(
                WARN,
                f"no ConfigRecord provided with `{self.clientapp_evaluate_config_key}` key ....",
            )
            record[self.clientapp_evaluate_config_key] = ConfigRecord(
                {"server_round": server_round}
            )
        else:
            # The user provided a `ConfigRecord` to send to `ClientApps`
            # when doing EVALUATE. Here we append the round number
            record[self.clientapp_evaluate_config_key]["server-round"] = server_round

        # Construct messages
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

    def evaluate(self, server_round: int, record: RecordDict) -> RecordDict:
        # Evaluate the current model parameters
        return {}

    def run(
        self, record_dict: RecordDict, grid: Grid, num_rounds: int, timeout: float
    ) -> MetricRecord:

        # do central eval with starting global parameters
        res = self.evaluate(server_round=0, record=record_dict)

        metrics_history: dict[int, dict[str, MetricRecord]] = {}

        for current_round in range(num_rounds):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round + 1, num_rounds)
            metrics_history[current_round] = {}

            # Configure train
            node_ids = self.sample_nodes(grid, is_train=True)
            messages = self.configure_train(current_round, record_dict, node_ids)
            # Communicate
            replies = grid.send_and_receive(messages, timeout=timeout)
            # aggregate fit
            res: RecordDict = self.aggregate_train(current_round, replies)
            # copy the content of the array records in res to those in record_dict
            for key, record in res.array_records.items():
                record_dict.array_records[key] = record
            # Log training metrics and append to history
            log(INFO, "Federated Training results: %s", res.metric_records)
            metrics_history[current_round]["train"] = res.metric_records

            # Configure evaluate
            node_ids = self.sample_nodes(grid, is_train=False)
            messages = self.configure_evaluate(current_round, record_dict, node_ids)
            # Communicate
            replies = grid.send_and_receive(messages, timeout=timeout)
            # Aggregate evaluate
            eval_res = self.aggregate_evaluate(current_round, replies)
            log(INFO, "Federated Evaluation results: %s", eval_res)
            metrics_history[current_round]["evaluate"] = eval_res

            # Centralised eval
            res = self.evaluate(server_round=current_round, record=record_dict)

        log(INFO, "Finished all rounds")
        log(INFO, "")

        return metrics_history
