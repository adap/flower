import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import math


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim1, hidden_dim2):
        super(VAE, self).__init__()

        # Encoder neural network
        self.encoder_hidden_1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_hidden_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.encoder_mean_logvar = nn.Linear(
            hidden_dim2, latent_dim * 2
        )  # *2 for mean and log-variance
        self.encoder_Lprime = nn.Linear(
            hidden_dim2, latent_dim * (latent_dim + 1) // 2
        )  # For L_prime

        # Decoder neural network
        self.decoder_hidden_1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_hidden_2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoder_output = nn.Linear(hidden_dim1, input_dim)

    def reparameterize(self, mu, L):
        eps = torch.randn_like(L)
        return torch.diag_embed(mu) + L * eps

    @staticmethod
    def lower_triangular_mask(latent_dim):
        mask = torch.tril(torch.ones(latent_dim, latent_dim))
        diag = torch.diag_embed(torch.ones(latent_dim))
        mask = mask - diag
        return mask.unsqueeze(0)  # Add a batch dimension

    # Function to convert Lprime to lower triangular matrix
    @staticmethod
    def Lprime_to_lower_triangular(Lprime, latent_dim):
        tril_indices = torch.tril_indices(row=latent_dim, col=latent_dim)
        L = torch.zeros(Lprime.shape[0], latent_dim, latent_dim)
        L[:, tril_indices[0], tril_indices[1]] = Lprime
        return L

    def forward(self, x):
        # Encode
        h = torch.relu(self.encoder_hidden_1(x))
        h = torch.relu(self.encoder_hidden_2(h))
        mu_logvar = self.encoder_mean_logvar(h)
        mu, logvar = torch.chunk(
            mu_logvar, 2, dim=1
        )  # Split into mean and log-variance
        Lprime = self.encoder_Lprime(h)
        sigma = torch.exp(0.5 * logvar)

        # Convert Lprime to lower triangular matrix
        Lprime_ = self.Lprime_to_lower_triangular(Lprime, latent_dim)

        # Apply the mask and add the diagonal of sigma
        L = self.lower_triangular_mask(latent_dim) * Lprime_ + torch.diag_embed(sigma)

        # Reparameterize
        # z = self.reparameterize(mu, L)
        eps = torch.randn_like(mu)
        z = mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)

        # Decode
        h = torch.relu(self.decoder_hidden_1(z))
        h = torch.relu(self.decoder_hidden_2(h))
        recon_x = self.decoder_output(h)

        return recon_x, mu, logvar, eps, z


def compute_elbo(x, vae):
    recon_x, mu, logvar, eps, z = vae(x)
    assert torch.all(torch.exp(logvar) >= 0), "var contains negative values"

    # Compute logqz
    logqz = -0.5 * torch.sum(
        (eps**2) + torch.log(torch.tensor(2 * math.pi)) + (logvar * 0.5), dim=1
    )
    
    # Compute logpz
    logpz = -0.5 * torch.sum((z**2) + torch.log(torch.tensor(2 * math.pi)), dim=1)

    # Compute logpx
    logpx = torch.sum(
        x * torch.log(torch.sigmoid(recon_x))
        + (1 - x) * torch.log(1 - torch.sigmoid(recon_x)),
        dim=1,
    )

    # Compute ELBO

    elbo = logpx + logpz - logqz
    return elbo.mean()


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = MNIST(root="./data", train=False, transform=transform)
# Instantiate VAE model

# Define data loader
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# Training function
def train_vae(vae, train_loader, optimizer, epochs):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # recon_x, mu, logvar = vae(x.view(-1, input_dim))
            elbo = compute_elbo(x.view(-1, input_dim), vae)
            loss = -elbo
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(
            "Epoch {}, Average Loss: {:.4f}".format(
                epoch + 1, total_loss / len(train_loader)
            )
        )


input_dim = 784  # MNIST image size 28x28=784
latent_dim = 2  # 2D latent space for visualization
hidden_dim1 = 256  # Hidden layer dimension
hidden_dim2 = 128  # Hidden layer dimension


vae = VAE(input_dim, latent_dim, hidden_dim1, hidden_dim2)

# Define optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)

# Train VAE
epochs = 10
train_vae(vae, train_loader, optimizer, epochs)

# Test VAE on test set
vae.eval()
with torch.no_grad():
    test_latents = []
    for x, _ in test_loader:
        recon_x, mu, logvar, eps, z = vae(x.view(-1, input_dim))
        test_latents.append(mu.numpy())

# Visualize latent representations in 2D
test_latents = np.concatenate(test_latents, axis=0)
plt.figure(figsize=(8, 6))
plt.scatter(test_latents[:, 0], test_latents[:, 1], c="b", marker="o", alpha=0.5)
plt.xlabel("Latent Variable 1")
plt.ylabel("Latent Variable 2")
plt.title("Latent Representations of Test Set")
plt.show()
