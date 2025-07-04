"""quickstart-catboost: A Flower / CatBoost app."""

import copy
import json
import random

from logging import INFO, WARN
from time import sleep

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp

# Create ServerApp
app = ServerApp()


def sample_nodes(grid: Grid, context: Context) -> list[int]:
    # Loop and wait until enough nodes are available.
    all_node_ids: list[int] = []
    sampled_node_ids: list[int] = []
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]
    while len(all_node_ids) < min_nodes:
        all_node_ids = list(grid.get_node_ids())
        if len(all_node_ids) >= min_nodes:
            # Sample nodes
            num_to_sample = int(len(all_node_ids) * fraction_sample)
            sampled_node_ids = random.sample(all_node_ids, num_to_sample)
            break
        log(INFO, "Waiting for nodes to connect...")
        sleep(2)

    log(INFO, "Sampled %s nodes (out of %s)", len(sampled_node_ids), len(all_node_ids))

    return sampled_node_ids


def construct_messages(
    node_ids: list[int],
    record: RecordDict,
    message_type: str,
    server_round: int,
) -> list[Message]:

    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=record,
            message_type=message_type,  # target method in ClientApp
            dst_node_id=node_id,
            group_id=str(server_round),
        )
        messages.append(message)
    return messages


def bagging_trees(model_dicts, global_model, server_round) -> bytes:
    """Perform bagging strategy on clients' models."""
    ind_s = 1 if server_round == 0 else 0
    for idx in range(ind_s, len(model_dicts)):
        global_model["oblivious_trees"].extend(model_dicts[idx]["oblivious_trees"])
    global_model = json.dumps(global_model).encode("utf-8")
    return global_model


@app.main()
def main(grid: Grid, context: Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]

    # Init global model
    global_model = b""  # init with empty list
    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Sample nodes
        node_ids = sample_nodes(grid, context)

        # Create messages
        gmodel_record = ConfigRecord({"model": global_model})
        record_dict = RecordDict({"gmodel": gmodel_record})
        messages = construct_messages(
            node_ids, record_dict, MessageType.TRAIN, server_round
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Collect received model and metric
        model_dicts = []
        auc_list = []
        for msg in replies:
            if msg.has_content():
                model_dicts.append(
                    json.loads(msg.content["metric_and_model"]["model_dict"])
                )
                auc_list.append(msg.content["metric_and_model"]["AUC"])
            else:
                log(WARN, f"message {msg.metadata.message_id} as an error.")

        # Load corresponding model based on server round
        global_model = (
            copy.deepcopy(model_dicts[0])
            if server_round == 0
            else json.loads(global_model)
        )

        # Perform bagging
        global_model = bagging_trees(model_dicts, global_model, server_round)

        # Log average eval AUC
        log(INFO, f"Avg eval AUC: {sum(auc_list)/len(auc_list):.3f}")
