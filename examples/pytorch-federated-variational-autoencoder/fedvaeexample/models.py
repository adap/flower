"""fedvae: A Flower app for Federated Variational Autoencoder."""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


class Net(nn.Module):
    def __init__(self, h_dim=576, z_dim=10) -> None:
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=2
            ),  # [batch, 6, 15, 15]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=2
            ),  # [batch, 16, 6, 6]
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar
