"""Adapted from the PyTorch Lightning quickstart example.

Source: pytorchlightning.ai (2021/02/04)
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


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


def load_data():
    # Training / validation set
    trainset = MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    trainset = Subset(trainset, range(1000))
    mnist_train, mnist_val = random_split(trainset, [800, 200])
    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(mnist_val, batch_size=32, shuffle=False, num_workers=0)

    # Test set
    testset = MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )
    testset = Subset(testset, range(10))
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def main() -> None:
    """Centralized training."""
    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Load model
    model = LitAutoEncoder()

    # Train
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
