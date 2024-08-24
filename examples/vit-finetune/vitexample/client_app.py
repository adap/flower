"""vitexample: A Flower / PyTorch app with Vision Transformers."""

import torch
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp


from vitexample.task import apply_train_transforms, get_dataset_partition
from vitexample.task import get_model, set_params, get_params, train


class FedViTClient(NumPyClient):
    def __init__(self, trainloader, learning_rate, num_classes):
        self.trainloader = trainloader
        self.learning_rate = learning_rate
        self.model = get_model(num_classes)

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Set optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Train locally
        avg_train_loss = train(
            self.model, self.trainloader, optimizer, epochs=1, device=self.device
        )
        # Return locally-finetuned part of the model
        return (
            get_params(self.model),
            len(self.trainloader.dataset),
            {"train_loss": avg_train_loss},
        )


def client_fn(context: Context):
    """Return a FedViTClient."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    dataset_name = context.run_config["dataset-name"]
    trainpartition = get_dataset_partition(num_partitions, partition_id, dataset_name)

    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    num_classes = context.run_config["num-classes"]
    trainset = trainpartition.with_transform(apply_train_transforms)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=2, shuffle=True
    )

    return FedViTClient(trainloader, lr, num_classes).to_client()


app = ClientApp(client_fn=client_fn)
