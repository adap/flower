

from logging import INFO, WARN
from time import sleep, time
from typing import Optional
from flwr.server import Grid
import numpy as np
from flwr.common import Message, ArrayRecord, RecordDict, ConfigRecord, log, MessageType, Array, MetricRecord


def aggregate(records: list[RecordDict], weighting_metric_name: str) -> RecordDict:
    # Aggregate the given records using the specified weighting metric
    aggregated = RecordDict()

    # Count total examples
    total_examples = 0
    examples_per_record = []
    for record in records:
        if isinstance(record, MetricRecord):
            if weighting_metric_name not in record:
                # alert user expected key is not found
                # default to homogeneous weighting (i.e. 1.0
                # for all model contributions)
                total_examples += 1.0
                examples_per_record.append(1.0)
            else:
                total_examples += record[weighting_metric_name]
                examples_per_record.append(record[weighting_metric_name])

    if total_examples == 0:
        # it could be that no `MetricRecord` was sent in the replies by ClientApps
        # Warn users
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
                        aggregated[name][key] = Array(value.numpy() * weight/total_examples)
                    else:
                        aggregated[name][key] = Array(aggregated[name][key].numpy() + value.numpy() * weight/total_examples)

            # For MetricRecord
            elif isinstance(record_item, MetricRecord):
                if name not in aggregated:
                    aggregated[name] = MetricRecord()
                # aggregate in-place
                for key, value in record_item.items():
                    if key not in aggregated[name]:
                        if isinstance(value, list):
                            aggregated[name][key] = (np.array(value) * weight/total_examples).tolist()
                        else:
                            aggregated[name][key] = value * weight/total_examples
                    else:
                        if isinstance(value, list):
                            aggregated[name][key] = (np.array(aggregated[name][key]) + np.array(value) * weight/total_examples).tolist()
                        else:
                            aggregated[name][key] += value *weight/total_examples

    return aggregated


def sample_nodes(grid: Grid, config: Optional[ConfigRecord], is_train: bool = False) -> list[int]:

    # TODO: if we want to decouple the sampling from the strategy, one idea is to control
    # TODO: the node samling stage via ConfigRecord. Howerver, this requires making some assumptions
    # TODO: in terms of which keys the users provide in the ConfigRecord. Hence the warning message below. 
    # TODO: this is one idea. If we had a class for sampling, things would be more explicit.
    if config is None:
        log(WARN, "No sampling config provided. Sampling all nodes...")
    #     log(WARN, "No ConfigRecord provided for sampling nodes. Defaulting to sample all nodes." \
    #     "Pass a {'sampling-config' : ConfigRecord('min-train-nodes': ..., 'fraction-train-nodes': ..." \
    #     "'min-evaluate-nodes': ..., 'fraction-evaluate-nodes': ...} to customize sampling.")

    min_nodes = 2
    all_node_ids: list[int] = []
    while len(all_node_ids) < min_nodes:
        all_node_ids = list(grid.get_node_ids())
        if len(all_node_ids) >= min_nodes:
            break
        log(INFO, "Waiting for nodes to connect...")
        sleep(2)
    
    log(
        INFO,
        f"configure_{'train' if is_train else 'evaluate'}: sampled {len(all_node_ids)} clients (out of {len(grid.get_node_ids())})",
    )

    return all_node_ids


class FedAvgA:

    def configure_train(
        self, server_round: int, record: RecordDict, node_ids: list[int]
    ) -> list[Message]:
        # Configure the next round of training

        if 'train-config' not in record:
            # warn once the user that no configRecord was found
            # with that specific key. Therefore a new ConfigRecord is going
            # to be created. If it was specified, a new entry to the configrecord
            # would be added simply injecting the server_round.
            log(WARN, "no ConfigRecord provided with `train-config` key ....")
            record['train-config'] = ConfigRecord({"server_round": server_round})
        else:
            # The user provided a `ConfigRecord` to send to `ClientApps`
            # when doing TRAIN. Here we append the round number
            record['train-config']['server-round'] = server_round

        # Construct messages
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=MessageType.TRAIN,  # target method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: list[Message],
    ) -> RecordDict:
        # Aggregate training results

        num_errors = 0
        for msg in replies:
            if msg.has_error():
                log(INFO, "Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error)
                num_errors += 1

        log(
            INFO,
            "aggregate_train: received %s results and %s failures",
            len(replies)-num_errors,
            num_errors,
        )

        rd = aggregate(records=[msg.content for msg in replies if msg.has_content()], weighting_metric_name="num-examples")

        return rd

    def configure_evaluate(
        self, server_round: int, record: RecordDict, node_ids: list[int]
    ) -> list[Message]:
        # Configure the next round of evaluation

        record['eval-config'] = ConfigRecord({"server_round": server_round})

        # Construct messages
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=MessageType.EVALUATE,  # target method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: list[Message],
    ) -> RecordDict:
        # Aggregate evaluation results
    
        num_errors = 0
        for msg in replies:
            if msg.has_error():
                log(INFO, "Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error)
                num_errors += 1

        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(replies)-num_errors,
            num_errors,
        )

        rd = aggregate(records=[msg.content for msg in replies if msg.has_content()], weighting_metric_name="num-examples")
        return rd

    def evaluate(
        self, server_round: int, record: RecordDict
    ) -> RecordDict:
        # Evaluate the current model parameters
        return {}




def run_strategy(record_dict: RecordDict, strategy, grid: Grid, num_rounds: int, timeout: float):

    # do central eval with starting global parameters
    res = strategy.evaluate(server_round=0, record=record_dict)

    for current_round in range(num_rounds):
        log(INFO, "")
        log(INFO, "Starting round %s/%s", current_round + 1, num_rounds)
        # Configure train
        node_ids = sample_nodes(grid, record_dict.get('sampling-config', None), is_train=True)
        messages = strategy.configure_train(current_round, record_dict, node_ids)
        # communicate
        replies = grid.send_and_receive(messages, timeout=timeout)
        # aggregate fit
        res: RecordDict = strategy.aggregate_train(current_round, replies)
        # copy the content of the array records in res to those in record_dict
        for key, record in res.array_records.items():
            record_dict.array_records[key] = record
        # So logging users are familiar with
        log(INFO, "Federated Training results: %s", res.metric_records)

        # Configure evaluate
        node_ids = sample_nodes(grid, record_dict.get('sampling-config', None))
        messages = strategy.configure_evaluate(current_round, record_dict, node_ids)
        # communicate
        replies = grid.send_and_receive(messages, timeout=timeout)
        # aggregate evaluate
        res = strategy.aggregate_evaluate(current_round, replies)
        log(INFO, "Federated Evaluation results: %s", res.metric_records)

        # Centralised eval
        res = strategy.evaluate(server_round=current_round, record=record_dict)