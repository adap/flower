"""bouquetfl: A Flower / PyTorch app."""

import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.client import ClientApp
from flwr.clientapp import ClientApp
from flwr.common import Code, Context

from bouquetfl.core.create_env import run_training_process_in_env

logger = logging.getLogger(__name__)

import torch

from bouquetfl import task

#####################################
########## Flower Client ############
#####################################

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
        ):
        print("Running training process in simulation mode")
        mode = "simulation"
    else:
        print("Running training process in real mode")
        mode = "real"

    # Save global model state dict locally for training process to load
    state_dict_global = msg.content["arrays"].to_torch_state_dict()
    torch.save(
        state_dict_global,
        f"/tmp/global_params_round_{msg.content['config']['server-round']}.pt",
    )

    # Run training process in restricted environment and get updated model state dict
    status, state_dict_updated = run_training_process_in_env(msg=msg, context=context, mode=mode)
    # If we ran into an OutOfMemoryError during training, we return a non-OK status and handle it in the server strategy by ignoring the update from this client for this round.
    if status.code != Code.OK:
        raise torch.OutOfMemoryError(
            f"{"\033[31m"}Client {context.node_config['partition-id']} has encountered an out-of-memory error{"\033[0m"}"
        )

    # Construct and return reply Message
    model_record = ArrayRecord(torch_state_dict=state_dict_updated)
    metrics = {
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local test data."""

    state_dict = msg.content["arrays"].to_torch_state_dict()

    testset = task.load_data(
        partition_id=context.node_config["partition-id"],
        num_clients=context.run_config["num-partitions"],
        num_workers=4,
        batch_size=context.run_config["batch-size"],
    )

    model = task.get_model()
    model.load_state_dict(state_dict)
    loss, accuracy = task.test(
        model=model,
        testloader=testset,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(
        f"{"\033[32m"}Global model evaluation on test set {context.node_config['partition-id']}{"\033[0m"}: accuracy: {round(100 * accuracy, 2)}%, loss: {round(loss, 2)}"
    )
    time.sleep(0.5)

    # Construct and return reply Message
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "num-examples": 1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
