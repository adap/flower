"""xgboost_quickstart: A Flower / XGBoost app."""

import json
from logging import INFO, WARN

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)

# Create ServerApp
app = ServerApp()


def aggregate_bagging(
    bst_prev_org: bytes | None,
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


def _construct_messages(
    node_ids: list[int],
    record: RecordDict,
    message_type: MessageType,
) -> list[Message]:

    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=record,
            message_type=message_type,  # target method in ClientApp
            dst_node_id=node_id,
        )
        messages.append(message)
    return messages

def configure_nodes(
    grid: Grid,
    fraction: float,
    min_available_nodes: int,
    global_model: bytes | None,
    server_round: int,
    message_type: str
) -> list[Message]:
    """Configure nodes for training or evaluation."""
    # Sample nodes
    sample_size = int(len(list(grid.get_node_ids())) * fraction)
    node_ids, num_total = sample_nodes(grid, min_available_nodes, sample_size)
    log(
        INFO,
        "configure_train: Sampled %s nodes (out of %s)",
        len(node_ids),
        len(num_total),
    )

    # Create messages
    gmodel_config_record = ConfigRecord({
        "model": global_model,
        "server-round": server_round,
    })
    recorddict = RecordDict({"model_config": gmodel_config_record})

    return _construct_messages(node_ids, recorddict, message_type)


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Load config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    min_available_nodes = context.run_config["min-available-nodes"]

    # Init global model
    global_model = b""  # Init with empty list

    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # --- TRAIN (CLIENTAPP-SIDE) ---------------------------------
        # Construct train messages
        messages = configure_nodes(
            grid,
            fraction_train,
            min_available_nodes,
            global_model,
            server_round,
            MessageType.TRAIN,
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Aggregate received models with bagging strategy
        for msg in replies:
            if msg.has_content():
                bst = msg.content["local_model"]["model"]
                global_model = aggregate_bagging(global_model, bst)
            else:
                log(WARN, f"message {msg.metadata.message_id} as an error.")

        # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
        # Construct evaluate messages
        messages = configure_nodes(
            grid,
            fraction_evaluate,
            min_available_nodes,
            global_model,
            server_round,
            MessageType.EVALUATE,
        )

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Aggregate received metrics
        replies_with_content = []
        for msg in replies:
            if msg.has_content():
                replies_with_content.append(msg.content)
            else:
                log(WARN, f"message {msg.metadata.message_id} as an error.")

        metrics =aggregate_metricrecords(
            replies_with_content,
            weighting_metric_name="num-examples",
        )

        # Log average eval AUC
        log(INFO, f"Avg eval AUC: {metrics['auc']:.3f}")
