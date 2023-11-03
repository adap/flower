import torch
import torch.nn as nn


class Generator(nn.Module):  # W_g
    def __init__(self, num_classes=10, latent_dim=20):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y):
        z0 = self.net(y)
        mu = self.fc_mu(z0)
        logvar = self.fc_logvar(z0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Enclassifier(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(Enclassifier, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.clf(z)
        return pred, mu, logvar
