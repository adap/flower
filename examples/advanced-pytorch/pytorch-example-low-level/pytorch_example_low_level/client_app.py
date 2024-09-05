"""pytorch-example-low-level: A low-level Flower / PyTorch app."""

import random
import time

import torch
from pytorch_example_low_level.task import Net, load_data, test, train
from pytorch_example_low_level.utils import (
    parameters_record_to_state_dict,
    state_dict_to_parameters_record,
)

from flwr.client import ClientApp
from flwr.common import ConfigsRecord, Context, Message, MetricsRecord, RecordSet

# Flower ClientApp
app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):

    # Initialize model
    model = Net()
    # Dynamically determine device (best for simulations)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this `ClientApp`'s dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Extract model received from `ServerApp`
    p_record = msg.content.parameters_records["global_model_record"]
    state_dict = parameters_record_to_state_dict(p_record)

    # apply to local PyTorch model
    model.load_state_dict(state_dict)

    # Get learning rate value sent from `ServerApp`
    lr = msg.content.configs_records["config"]["lr"]
    # Train with local dataset
    train_loss = train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        lr=lr,
        device=device,
    )

    # Put resulting model into a ParametersRecord
    p_record = state_dict_to_parameters_record(model.state_dict())

    # Send reply back to `ServerApp`
    reply_content = RecordSet()
    reply_content.parameters_records["updated_model_dict"] = p_record
    # Return message
    return msg.create_reply(reply_content)


@app.evaluate()
def eval_fn(msg: Message, context: Context):

    # Initialize model
    model = Net()
    # Dynamically determine device (best for simulations)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this `ClientApp`'s dataset
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, evalloader = load_data(partition_id, num_partitions)

    # Extract model received from `ServerApp`
    p_record = msg.content.parameters_records["global_model_record"]
    state_dict = parameters_record_to_state_dict(p_record)

    # apply to local PyTorch model
    model.load_state_dict(state_dict)

    # Evaluate with local dataset
    loss, accuracy = test(
        model,
        evalloader,
        device=device,
    )

    # Put resulting metrics into a MetricsRecord
    m_record = MetricsRecord({"loss": loss, "accuracy": accuracy})

    # Send reply back to `ServerApp`
    reply_content = RecordSet()
    reply_content.metrics_records["clientapp-evaluate"] = m_record
    # Return message
    return msg.create_reply(reply_content)


@app.query()
def query(msg: Message, context: Context):
    """A basic query method that aims to exemplify some opt-in functionality.

    The node running this `ClientApp` reacts to an incomming message by returning
    a `True` or a `False`. If `True`, this node will be sampled by the `ServerApp`
    to receive the global model and do evaluation in its `@app.eval()` method.
    """

    # Inspect message
    c_record = msg.content.configs_records["query-config"]
    # print(f"Received: {c_record = }")

    # Sleep for a random amount of time, will result in some nodes not
    # repling back to the `ServerApp` in time
    time.sleep(random.randint(0, 2))

    # Randomly set True or False as opt-in in the evaluation stage
    # Note the keys used for the records below are arbitrary, but both `ServerApp`
    # and `ClientApp` need to be aware of them.
    c_record_response = ConfigsRecord(
        {"opt-in": random.random() > 0.5, "ts": time.time()}
    )
    reply_content = RecordSet(configs_records={"query-response": c_record_response})

    return msg.create_reply(content=reply_content)
