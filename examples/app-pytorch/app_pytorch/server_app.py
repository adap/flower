"""app-pytorch: A Flower / PyTorch app."""

import random
from logging import INFO, WARN
from time import sleep

import torch
from app_pytorch.task import Net

from flwr.common import ArrayRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]

    # Init global model
    global_model = Net()
    global_model_key = "model"

    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                num_to_sample = int(len(all_node_ids) * fraction_sample)
                node_ids = random.sample(all_node_ids, num_to_sample)
                break
            log(INFO, "Waiting for nodes to connect...")
            sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        # Create messages
        gmodel_record = ArrayRecord(global_model.state_dict())
        recorddict = RecordDict({global_model_key: gmodel_record})
        messages = construct_messages(
            node_ids, recorddict, MessageType.TRAIN, server_round
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Convert Parameter Records in messages back to PyTorch's state_dicts
        state_dicts = []
        avg_train_losses = []
        for msg in replies:
            if msg.has_content():
                state_dicts.append(msg.content[global_model_key].to_torch_state_dict())
                avg_train_losses.append(msg.content["train_metrics"]["train_loss"])
            else:
                log(WARN, f"message {msg.metadata.message_id} as an error.")

        # Compute average state dict
        avg_statedict = average_state_dicts(state_dicts)
        # Materialize global model
        global_model.load_state_dict(avg_statedict)

        # Log average train loss
        log(INFO, f"Avg train loss: {sum(avg_train_losses)/len(avg_train_losses):.3f}")

        ## Start evaluate round

        # Sample all nodes
        all_node_ids = grid.get_node_ids()
        gmodel_record = ArrayRecord(gmodel_record.to_torch_state_dict())
        recorddict = RecordDict({global_model_key: gmodel_record})
        messages = construct_messages(
            node_ids, recorddict, MessageType.EVALUATE, server_round
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Aggregate evaluate losss
        avg_eval_acc = []
        for msg in replies:
            if msg.has_content():
                avg_eval_acc.append(msg.content["eval_metrics"]["eval_acc"])
            else:
                log(WARN, f"message {msg.metadata.message_id} as an error.")

        # Log average train loss
        log(INFO, f"Avg eval acc: {sum(avg_eval_acc)/len(avg_eval_acc):.3f}")


def construct_messages(
    node_ids: list[int],
    record: RecordDict,
    message_type: MessageType,
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
