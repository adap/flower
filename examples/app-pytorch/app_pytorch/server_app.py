"""app-pytorch: A Flower / PyTorch app."""

import random
from logging import INFO

from app_pytorch.task import Net

import flwr as fl
from flwr.common.logger import log

from .things_we_need import *

# Create ServerApp
app = fl.ServerApp()

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


@app.main()
def main(driver: fl.Driver, context: fl.Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]
    fraction_sample = context.run_config["fraction-sample"]
    min_nodes = 2

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
