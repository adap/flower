"""pytorch-example-low-level: A low-level Flower / PyTorch app."""

import torch
from pytorch_example_low_level.task import Net, load_data, train
from pytorch_example_low_level.utils import (
    parameters_record_to_state_dict,
    state_dict_to_parameters_record,
)

from flwr.client import ClientApp
from flwr.common import Context, Message, RecordSet
from flwr.common.logger import log

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
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Extract model received from `ServerApp`
    p_record = msg.content.parameters_records["global_model_record"]
    state_dict = parameters_record_to_state_dict(p_record)

    # apply to local PyTorch model
    model.load_state_dict(state_dict)

    # Train with local dataset
    train_loss = train(
        model,
        trainloader,
        context.run_config["local-epochs"],
        lr=0.1,
        device=device,
    )

    # Put resulting model into a ParametersRecord
    p_record = state_dict_to_parameters_record(model.state_dict())

    # Send reply back to `ServerApp`
    reply_content = RecordSet()
    reply_content.parameters_records["updated_model_dict"] = p_record
    # Return message
    return msg.create_reply(reply_content)
