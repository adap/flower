"""Adapted from the PyTorch Lightning quickstart example.

Source: pytorchlightning.ai (2021/02/04)
"""

from flwr_datasets import FederatedDataset
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage=None):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)


def collate_fn(batch):
    """Change the dictionary to tuple to keep the exact dataloader behavior."""
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [transforms.functional.to_tensor(img) for img in batch["image"]]
    return batch


def load_data(partition):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    partition = fds.load_partition(partition, "train")

    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        collate_fn=collate_fn,
        num_workers=1,
    )
    testloader = DataLoader(
        partition_full["test"], batch_size=32, collate_fn=collate_fn, num_workers=1
    )
    return trainloader, valloader, testloader


def main() -> None:
    """Centralized training."""

    # Load data
    train_loader, val_loader, test_loader = load_data(0)

    # Load model
    model = LitAutoEncoder()

    # Train
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
