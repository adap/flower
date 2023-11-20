# %%
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDENTIFIER = "vae_single"
if not os.path.exists(IDENTIFIER):
    os.makedirs(IDENTIFIER)
bs = 64
# MNIST Dataset
train_dataset = datasets.MNIST(
    root="./mnist_data/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./mnist_data/", train=False, transform=transforms.ToTensor(), download=False
)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=bs, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=bs, shuffle=False
)


# %%
class VAE(nn.Module):
    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=2):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


# build model
vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

# %%
vae

# %%
optimizer = optim.Adam(vae.parameters())


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


# %%
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )
    return vae


# %%
def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def test2(net, testloader, device, rnd=None, folder=None):
    """Validate the network on the entire test set."""
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images = data[0].to(device)
            break
        save_image(images.view(64, 1, 28, 28), f"{folder}/true_img_at_{rnd}.png")

        # Generate image using your generate function
        generated_tensors = generate(net, images)
        generated_img = generated_tensors[0]
        save_image(
            generated_img.view(64, 1, 28, 28), f"{folder}/test_generated_at_{rnd}.png"
        )


def visualize_latent_representation(model, test_loader, device, rnd=None, folder=None):
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Apply PCA using PyTorch
    cov_matrix = torch.tensor(np.cov(all_latents.T), dtype=torch.float32)
    _, _, V = torch.svd_lowrank(cov_matrix, q=2)

    # Project data onto the first two principal components
    reduced_latents = torch.mm(torch.tensor(all_latents, dtype=torch.float32), V)

    # Convert to numpy array
    reduced_latents = reduced_latents.numpy()

    # Visualize latent representation
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_latents[:, 0], reduced_latents[:, 1], c=all_labels, cmap="tab10"
    )
    plt.colorbar(scatter, label="Digit Label")
    plt.title("Latent Representation Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig(f"{folder}/latent_rep_at_{rnd}.png")


# %%
for epoch in range(1, 51):
    model = train(epoch)
    test()
    test2(model, test_loader, device, epoch, IDENTIFIER)
    visualize_latent_representation(model, test_loader, device, epoch, IDENTIFIER)


# %%
with torch.no_grad():
    z = torch.randn(64, 2).cuda()
    sample = vae.decoder(z).cuda()

    save_image(sample.view(64, 1, 28, 28), "single_app_cpu/sample_" + ".png")
