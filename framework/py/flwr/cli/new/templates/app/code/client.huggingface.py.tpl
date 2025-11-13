"""$project_name: A Flower / $framework_str app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from transformers import AutoModelForSequenceClassification

from $import_name.task import load_data
from $import_name.task import test as test_fn
from $import_name.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, _ = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Initialize it with the received weights
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Train the model on local data
    train_loss = train_fn(
        net,
        trainloader,
        context.run_config["local-steps"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(net.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    _, valloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Initialize it with the received weights
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Evaluate the model on local data
    val_loss, val_accuracy = test_fn(
        net,
        valloader,
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(net.state_dict())
    metrics = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
