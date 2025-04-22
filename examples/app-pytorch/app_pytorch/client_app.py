"""app-pytorch: A Flower / PyTorch app."""

import torch
from app_pytorch.task import Net, load_data
from app_pytorch.task import test as test_fn
from app_pytorch.task import train as train_fn

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

# Flower ClientApp
app = ClientApp()


@app.evaluate()
def evaluate(msg: Message, context: Context):

    # Prepare
    model, device, data_loader = setup_client(msg, context, is_train=False)

    # Local evaluation
    _, eval_acc = test_fn(
        model,
        data_loader,
        device,
    )

    # Construct reply
    metric_record = MetricRecord({"eval_acc": eval_acc})
    content = RecordDict({"eval_metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.train()
def train(msg: Message, context: Context):

    # Prepare
    model, device, data_loader = setup_client(msg, context, is_train=True)

    # Local training
    local_epochs = context.run_config["local-epochs"]
    train_loss = train_fn(
        model,
        data_loader,
        local_epochs,
        device,
    )

    # Extract state_dict from model and construct reply message
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord({"train_loss": train_loss})
    content = RecordDict({"model": model_record, "train_metrics": metric_record})
    return Message(content=content, reply_to=msg)


def setup_client(msg: Message, context: Context, is_train: bool):

    # Instantiate model
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Apply global model weights from message
    model.load_state_dict(msg.content["model"].to_torch_state_dict())
    model.to(device)

    # Load partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    return model, device, trainloader if is_train else valloader
