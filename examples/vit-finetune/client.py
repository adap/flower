import torch
from torch.utils.data import DataLoader

import flwr
from flwr.client import NumPyClient
from dataset import apply_transforms, get_dataset_with_partitions
from model import get_model, set_parameters, train


class FedViTClient(NumPyClient):
    def __init__(self, trainset):
        self.trainset = trainset
        self.model = get_model()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_for_finetuning(self):
        """Freeze all parameter except those in the final head.

        Only output MLP will be updated by the client and therefore, the only part of
        the model that will be federated (hence, communicated back to the server for
        aggregation.)
        """

        # Disable gradients for everything
        self.model.requires_grad_(False)
        # Now enable just for output head
        self.model.heads.requires_grad_(True)

    def get_parameters(self, config):
        """Get locally updated parameters."""
        finetune_layers = self.model.heads
        return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # Get some info from the config
        # Get batchsize and LR set from server
        batch_size = config["batch_size"]
        lr = config["lr"]

        trainloader = DataLoader(
            self.trainset, batch_size=batch_size, num_workers=2, shuffle=True
        )

        # Set optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Train locally
        avg_train_loss = train(
            self.model, trainloader, optimizer, epochs=1, device=self.device
        )
        # Return locally-finetuned part of the model
        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {"train_loss": avg_train_loss},
        )


# Downloads and partition dataset
federated_ox_flowers, _ = get_dataset_with_partitions(num_partitions=20)


def client_fn(cid: str):
    """Return a FedViTClient that trains with the cid-th data partition."""

    trainset_for_this_client = federated_ox_flowers.load_partition(int(cid), "train")

    trainset = trainset_for_this_client.with_transform(apply_transforms)

    return FedViTClient(trainset).to_client()


# To be used with Flower Next
app = flwr.client.ClientApp(
    client_fn=client_fn,
)
