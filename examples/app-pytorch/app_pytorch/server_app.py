"""app-pytorch: A Flower / PyTorch app."""

import random
from abc import abstractmethod
from logging import INFO
from typing import Callable

import torch
from app_pytorch.task import Net

import flwr as fl
from flwr.common.logger import log

# Create ServerApp
app = fl.ServerApp()


@app.main()
def main(driver: fl.Driver, context: fl.Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]
    fraction_sample = context.run_config["fraction-sample"]
    min_nodes = 2

    # Init global model
    global_model = Net()
    global_model_key = "model"

    # Init aggregators
    train_aggregator = SequentialAggregator(
        [
            ParametersAggregator(
                record_key=global_model_key,
                weight_factor_key=lambda rs: rs["train_metrics"]["num_examples"],
            ),
            MetricsAggregator(
                record_key="train_metrics",
                aggregate_key="train_loss",
            ),
        ]
    )
    eval_aggregator = MetricsAggregator(record_key="eval_metrics")

    for server_round in range(1, num_rounds + 1):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round, num_rounds)

        # Loop and wait until enough nodes are available.
        log(INFO, "Waiting for nodes to connect...")
        all_node_ids = driver.get_node_ids(min_num_nodes=min_nodes)

        # Sample nodes
        num_to_sample = int(len(all_node_ids) * fraction_sample)
        node_ids = random.sample(all_node_ids, num_to_sample)
        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        # Create messages
        gmodel_record = fl.ParametersRecord(global_model.state_dict())
        recordset = fl.RecordSet(parameters_records={global_model_key: gmodel_record})
        messages = create_broadcast_messages(
            driver, recordset, fl.MessageType.TRAIN, node_ids, str(server_round)
        )

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Convert Parameter Records in messages back to PyTorch's state_dicts
        agg_rs = train_aggregator(replies)

        # Materialize global model
        global_model.load_state_dict(agg_rs[global_model_key].to_state_dict())

        # Log average train loss
        log(INFO, f"Avg train loss: {agg_rs['train_metrics']['train_loss']:.3f}")

        ## Start evaluate round

        # Sample all nodes
        all_node_ids = driver.get_node_ids()
        log(INFO, "Sampled %s nodes (out of %s)", len(all_node_ids), len(all_node_ids))
        recordset = fl.RecordSet({global_model_key: fl.ParametersRecord(global_model)})
        messages = create_broadcast_messages(
            driver, recordset, fl.MessageType.EVALUATE, all_node_ids, str(server_round)
        )

        # Send messages and wait for all results
        replies = driver.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Aggregate evaluate losss
        agg_rs = eval_aggregator(replies)

        # Log average train loss
        log(INFO, f"Avg eval acc: {agg_rs['eval_metrics']['eval_acc']:.3f}")
        log(INFO, f"Avg eval loss: {agg_rs['eval_metrics']['eval_loss']:.3f}")


def create_broadcast_messages(
    driver: fl.Driver,
    record: fl.RecordSet,
    message_type: fl.MessageType,
    node_ids: list[int],
    group_id: str,
) -> list[fl.Message]:

    messages = []
    for node_id in node_ids:  # one message for each node
        # Copy non-parameters records to avoid modifying the original record
        new_record = fl.RecordSet(
            parameters_records=record.parameters_records,
            metrics_records={k: v.copy() for k, v in record.metrics_records.items()},
            configs_records={k: v.copy() for k, v in record.configs_records.items()},
        )
        message = driver.create_message(
            content=new_record,
            message_type=message_type,  # target method in ClientApp
            dst_node_id=node_id,
            group_id=group_id,
        )
        messages.append(message)
    return messages


def average_state_dicts(state_dicts):
    """Return average state_dict."""
    # Initialize the averaged state dict
    avg_state_dict = {}

    # Iterate over keys in the first state dict
    for key in state_dicts[0]:
        # Stack all the tensors for this parameter across state dicts
        stacked_tensors = torch.stack([sd[key] for sd in state_dicts])
        # Compute the mean across the 0th dimension
        avg_state_dict[key] = torch.mean(stacked_tensors, dim=0)

    return avg_state_dict


class BaseAggregator:

    def __call__(self, messages: list[fl.Message]) -> fl.RecordSet:
        return self.aggregate(messages)

    @abstractmethod
    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...


class SequentialAggregator(BaseAggregator):

    def __init__(
        self,
        aggregators: list[BaseAggregator],
    ) -> None: ...


class ParametersAggregator(BaseAggregator):

    def __init__(
        self,
        record_key: str | None = None,
        weight_factor_key: Callable[[fl.RecordSet], float | int] | None = None,
        aggregate_key: list[str] | str = "*",
        reduction: str = "mean",
    ) -> None: ...

    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...


class MetricsAggregator(BaseAggregator):

    def __init__(
        self,
        record_key: str | None = None,
        weight_factor_key: Callable[[fl.RecordSet], float | int] | str | None = None,
        aggregate_key: list[str] | str = "*",
        reduction: str = "mean",
    ) -> None: ...

    def aggregate(self, messages: list[fl.Message]) -> fl.RecordSet: ...
