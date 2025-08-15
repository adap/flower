

from logging import INFO
from time import sleep
from typing import Optional
from flwr.server import Grid
import numpy as np
from flwr.common import Message, ArrayRecord, RecordDict, ConfigRecord, log, MessageType, Array, MetricRecord
from .strategy import Strategy


class FedAvg(Strategy):
    def initialize_parameters(self, grid: Grid) -> Optional[ArrayRecord]:
        # Initialize global model parameters
        return None

    def configure_train(
        self, server_round: int, record: RecordDict, grid: Grid
    ) -> list[Message]:
        # Configure the next round of training

        record['train-config'] = ConfigRecord({"server_round": server_round})

        # Sample nodes
        min_nodes = 2
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_nodes:
                break
            log(INFO, "Waiting for nodes to connect...")
            sleep(2)
        log(INFO, "Sampled %s nodes", len(all_node_ids))

        # Construct messages
        messages = []
        for node_id in all_node_ids:  # one message for each node
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
        results: list[Message],
        failures: list[Message],
    ) -> RecordDict:
        # Aggregate training results

        log(INFO, "Received %s results and %s failures", len(results), len(failures))

        rd: RecordDict = results[0].content
        
        # To perform averaging
        # TODO: implement as dict so each entry in final recorddict
        # TODO: is averaged independently
        counter_ar = 0
        counter_metrics = 0

        for msg in results[1:]:
            for name, record in msg.content.items():
                if isinstance(record, ArrayRecord):
                    counter_ar += 1
                    for arr_name, array in record.items():
                        # TODO: optimize (currently low memory usage, but repeated serde)
                        rd[name][arr_name] = Array(rd[name][arr_name].numpy() + array.numpy())
                    
                elif isinstance(record, MetricRecord):
                    counter_metrics += 1
                    for metric_name, metric_value in record.items():
                        if isinstance(metric_value, list):
                            # TODO: optimize
                            rd[name][metric_name] = (np.array(rd[name][metric_name]) + np.array(metric_value)).tolist()
                        else:
                            rd[name][metric_name] += metric_value

        # now divide the value of all arrayrecords and metric records by counter
        for name, record in rd.items():
            # if array record, average all arrays by counter
            if isinstance(record, ArrayRecord):
                for arr_name, array in record.items():
                    rd[name][arr_name] = Array(array.numpy() / counter_ar)

            # elif metric record average all entries by counter
            elif isinstance(record, MetricRecord):
                for metric_name, metric_value in record.items():
                    if isinstance(metric_value, list):
                        rd[name][metric_name] = (np.array(rd[name][metric_name]) / counter_metrics).tolist()
                    else:
                        rd[name][metric_name] /= counter_metrics

        return rd

    def configure_evaluate(
        self, server_round: int, record: RecordDict, grid: Grid
    ) -> list[Message]:
        # Configure the next round of evaluation
        record['eval-config'] = ConfigRecord({"server_round": server_round})

        # Sample nodes
        min_nodes = 2
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_nodes:
                break
            log(INFO, "Waiting for nodes to connect...")
            sleep(2)
        
        log(INFO, "Sampled %s nodes", len(all_node_ids))

        # Construct messages
        messages = []
        for node_id in all_node_ids:  # one message for each node
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
        results: list[Message],
        failures: list[Message],
    ) -> RecordDict:
        # Aggregate evaluation results
    
        log(INFO, "Received %s results and %s failures", len(results), len(failures))

        rd: RecordDict = results[0].content
        
        # To perform averaging
        # TODO: implement as dict so each entry in final recorddict
        # TODO: is averaged independently
        counter = 0

        for msg in results[1:]:
            for name, record in msg.content.items():
                if isinstance(record, MetricRecord):
                    counter += 1
                    for metric_name, metric_value in record.items():
                        if isinstance(metric_value, list):
                            # TODO: optimize
                            rd[name][metric_name] = (np.array(rd[name][metric_name]) + np.array(metric_value)).tolist()
                        else:
                            rd[name][metric_name] += metric_value

        # now divide the value of all arrayrecords and metric records by counter
        for name, record in rd.items():
            # if array record, average all arrays by counter
            if isinstance(record, MetricRecord):
                for metric_name, metric_value in record.items():
                    if isinstance(metric_value, list):
                        rd[name][metric_name] = (np.array(rd[name][metric_name]) / counter).tolist()
                    else:
                        rd[name][metric_name] /= counter

        return rd

    def evaluate(
        self, server_round: int, record: RecordDict
    ) -> RecordDict:
        # Evaluate the current model parameters
        return {}
    

def run_strategy(record_dict: RecordDict, strategy: Strategy, grid: Grid, num_rounds: int, timeout: float):

    # do central eval with starting global parameters
    res = strategy.evaluate(server_round=0, record=record_dict)

    for current_round in range(num_rounds):
        log(INFO, "Starting round %s/%s", current_round + 1, num_rounds)
        # prepare fit
        messages = strategy.configure_fit(current_round, record_dict, grid)
        # communicate
        replies = grid.send_and_receive(messages, timeout=timeout)
        # split results and failures in two lists
        results, failures = [], []
        for reply in replies:
            if reply.has_content():
                results.append(reply)
            else:
                failures.append(reply)
        # aggregate fit
        res = strategy.aggregate_fit(current_round, replies, failures)
        # copy the content of the array records in res to those in record_dict
        for key, record in res.array_records.items():
            record_dict.array_records[key] = record
        # So logging users are familiar with
        log(INFO, "Federated Training results: %s", res.metric_records)

        # prepare evaluate
        messages = strategy.configure_evaluate(current_round, record_dict, grid)
        # communicate
        replies = grid.send_and_receive(messages, timeout=timeout)
        # split results and failures in two lists
        results, failures = [], []
        for reply in replies:
            if reply.has_content():
                results.append(reply)
            else:
                failures.append(reply)
        # aggregate evaluate
        res = strategy.aggregate_evaluate(current_round, results, failures)
        log(INFO, "Federated Evaluation results: %s", res.metric_records)

        # Centralised eval
        res = strategy.evaluate(server_round=current_round, record=record_dict)