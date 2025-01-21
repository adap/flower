"""app-pytorch: A Flower / PyTorch app."""

import torch
from app_pytorch.task import Net, load_data
from app_pytorch.task import test as test_fn
from app_pytorch.task import train as train_fn

import flwr as fl

# Flower ClientApp
app = fl.ClientApp()

# Instantiate model
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.evaluate()
def evaluate(msg: fl.Message, context: fl.Context):

    # Prepare
    model.load_state_dict(msg.content["model"].to_state_dict())
    model.to(device)
    data_loader = get_dataloader("test", context)

    # Local evaluation
    eval_loss, eval_acc = test_fn(model, data_loader, device)

    # Construct reply
    content = fl.RecordSet()
    content["eval_metrics"] = fl.MetricsRecord(
        {"eval_loss": eval_loss, "eval_acc": eval_acc}
    )
    return msg.create_reply(content=content)


@app.train()
def train(msg: fl.Message, context: fl.Context):

    # Prepare
    model.load_state_dict(msg.content["model"].to_state_dict())
    model.to(device)
    data_loader = get_dataloader("train", context)

    # Local training
    local_epochs = context.run_config["local-epochs"]
    train_loss = train_fn(model, data_loader, local_epochs, device)

    # Extract state_dict from model and construct reply message
    content = fl.RecordSet()
    content["model"] = fl.ParametersRecord(model.state_dict())
    content["train_metrics"] = fl.MetricsRecord({"train_loss": train_loss})
    return msg.create_reply(content=content)


def get_dataloader(split: str, context: fl.Context):
    """Return train or test dataloader."""
    # Load partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader = load_data(partition_id, num_partitions)

    if split == "train":
        return trainloader
    if split == "test":
        return testloader
    raise ValueError(f"Invalid split: {split}")
