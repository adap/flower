"""vitpoultry: A Flower / PyTorch app with Vision Transformers for Poultry Health."""

import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from vitpoultry.task import (
    apply_train_transforms,
    get_dataset_partition,
    get_model,
    load_local_data,
    trainer,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    num_classes = context.run_config["num-classes"]

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # Simulation Engine: use flwr_datasets and partition data on the fly
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        dataset_name = context.run_config["dataset-name"]
        trainpartition = get_dataset_partition(
            num_partitions, partition_id, dataset_name
        )
        trainset = trainpartition.with_transform(apply_train_transforms)
    else:
        # Deployment Engine: load data from a local path on the SuperNode
        data_path = context.node_config["data-path"]
        trainset = load_local_data(data_path, apply_train_transforms)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=2, shuffle=True
    )

    model = get_model(num_classes)
    finetune_layers = model.heads
    finetune_layers.load_state_dict(
        msg.content["arrays"].to_torch_state_dict(), strict=True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    avg_train_loss = trainer(model, trainloader, optimizer, epochs=1, device=device)

    model_record = ArrayRecord(finetune_layers.state_dict())
    metrics = {
        "train_loss": avg_train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


