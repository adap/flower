import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_mnist import (
    test,
    visualize_gen_image,
    visualize_gmm_latent_representation,
    non_iid_train_iid_test_6789,
    subset_alignment_dataloader,
    alignment_dataloader,
    train_align,
    eval_reconstrution,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import plotly.graph_objects as go
from torchvision.utils import save_image
from torch.utils.data import DataLoader

NUM_CLIENTS = 2
NUM_CLASSES = 10
samples_per_class = 200
sub_alignment_dataloader = subset_alignment_dataloader(
    samples_per_class=samples_per_class,
    batch_size=samples_per_class * NUM_CLASSES,
    shuffle=True,
)
new_batch_size = 64  # Specify your desired batch size
new_testloader = DataLoader(
    sub_alignment_dataloader.dataset, batch_size=new_batch_size, shuffle=True
)

full_alignment_dataloader = alignment_dataloader(
    samples_per_class=samples_per_class,
    batch_size=samples_per_class * NUM_CLASSES,
    shuffle=True,
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate(net, image, label):
    """Reproduce the input with trained infoVAE."""
    with torch.no_grad():
        return net.forward(image, label)


def visualize_gen_image(net, testloader, device, rnd=None, folder=None):
    """Validate the network on the entire test set."""
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(device)
            labels = data[1].to(device)

            break
        save_image(images.view(64, 1, 28, 28), f"{folder}/true_img_at_{rnd}.png")

        # Generate image using your generate function
        generated_tensors = generate(net, images, labels)
        generated_img = generated_tensors[0]
        save_image(
            generated_img.view(64, 1, 28, 28), f"{folder}/test_generated_at_{rnd}.png"
        )
        return (
            f"{folder}/true_img_at_{rnd}.png",
            f"{folder}/test_generated_at_{rnd}.png",
        )


class infoVAE(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size1=256,
        hidden_size2=128,
        latent_size=2,
        num_classes=10,
        dis_hidden_size=4,
    ):
        super(infoVAE, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_size2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size2, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid(),
        )

        # Discriminator for InfoGAN
        self.discriminator = nn.Sequential(
            nn.Linear(latent_size, dis_hidden_size),
            nn.ReLU(),
            nn.Linear(dis_hidden_size, num_classes),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, y):
        y = F.one_hot(y, self.num_classes)
        zy = torch.cat((z, y), dim=1)
        return self.decoder(zy)

    def discriminate(self, z):
        return self.discriminator(z)

    def forward(self, x, y):
        z, mu, logvar = self.encode(x.view(-1, 784))
        recon_x = self.decode(z, y)
        pred_y = self.discriminate(z)

        return recon_x, mu, logvar, pred_y


def info_vae_loss(recon_x, x, mu, logvar, y, y_pred, lambda_kl=0.01, lambda_info=1.0):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction="sum"
    )  # Adjust for your data type

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Information loss (cross-entropy between predicted and true auxiliary variable)
    info_loss = F.cross_entropy(y_pred, y, reduction="sum")

    # Total loss
    total_loss = recon_loss + lambda_kl * kl_loss + lambda_info * info_loss

    return total_loss, recon_loss.item(), kl_loss.item(), info_loss.item()


sweep_config = {
    "method": "random",
    "metric": {"name": "total_loss", "goal": "minimize"},
    "parameters": {
        # "lambda_kl": {"distribution": "uniform", "min": 0.01, "max": 1},
        "lambda_kl": {"values": [1, 0.01, 0.1]},
        # "lambda_info": {"distribution": "uniform", "min": 0.1, "max": 1},
        "lambda_info": {"values": [5, 10]},
        "dis_hidden_size": {"values": [8, 4]},
    },
}
sweep_id = wandb.sweep(sweep=sweep_config, project="info-vae")


def train():
    run = wandb.init(
        entity="mak",
        group="info-vae_with_kl_4cls",
        reinit=True,
    )

    print(f"running these hparams-> {wandb.config}")
    ref_model = infoVAE(dis_hidden_size=wandb.config["dis_hidden_size"]).to(DEVICE)
    # ref_model = infoVAE(dis_hidden_size=4).to(DEVICE)
    opt_ref = torch.optim.Adam(ref_model.parameters(), lr=1e-3)
    # Training loop with W&B logging
    for ep in range(5000):
        for images, labels in sub_alignment_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            opt_ref.zero_grad()
            recon_images, mu, logvar, y_pred = ref_model(images, labels)
            total_loss, recon_loss, kl_loss, info_loss = info_vae_loss(
                recon_images,
                images,
                mu,
                logvar,
                labels,
                y_pred,
                # 0.01,
                # 1,
                wandb.config["lambda_kl"],
                wandb.config["lambda_info"],
            )
            total_loss.backward()
            opt_ref.step()

        if ep % 100 == 0:
            print(
                f"Epoch {ep}, Loss {total_loss.item()}, Recon Loss {recon_loss}, KL Loss {kl_loss}, Info Loss {info_loss}"
            )
            print("--------------------------------------------------")

        # Log latent representations plot to W&B
        ref_model.eval()
        with torch.no_grad():
            test_latents = []
            test_labels = []  
            for (
                x,
                labels,
            ) in sub_alignment_dataloader:  # Retrieve labels from test_loader
                x = x.to(DEVICE)
                labels = labels.to(DEVICE)
                recon_images, mu, logvar, _ = ref_model(x, labels)
                test_latents.append(mu.cpu().numpy())
                test_labels.append(labels.cpu().numpy())  # Store labels

            # Visualize latent representations in 2D
            test_latents = np.concatenate(test_latents, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)  # Concatenate all labels

            # Create traces for each class
            traces = []
            for label in np.unique(test_labels):
                indices = np.where(test_labels == label)
                trace = go.Scatter(
                    x=test_latents[indices, 0].flatten(),
                    y=test_latents[indices, 1].flatten(),
                    mode="markers",
                    name=str(label),
                    marker=dict(size=8),
                )
                traces.append(trace)

            # Create layout
            layout = go.Layout(
                title="Latent Space Visualization of Test Set with Class Highlight",
                xaxis=dict(title="Latent Dimension 1"),
                yaxis=dict(title="Latent Dimension 2"),
                legend=dict(title="Class"),
            )

            # Create figure
            fig = go.Figure(data=traces, layout=layout)
            # plt.savefig("info_vae_latent_space_visualization.png")

            # Log metrics to W&B
            wandb.log(
                {
                    "epoch": ep,
                    "total_loss": total_loss.item(),
                    "recon_loss": recon_loss,
                    "kl_loss": kl_loss,
                    "info_loss": info_loss,
                    "latent_space_visualization": fig,
                },
                step=ep,
            )
            true_img, gen_img = visualize_gen_image(
                ref_model, new_testloader, DEVICE, rnd=ep, folder="tst_img_gen"
            )
            wandb.log(
                {
                    "true_img": wandb.Image(true_img),
                    "gen_img": wandb.Image(gen_img),
                },
                step=ep,
            )

    wandb.finish()


if __name__ == "__main__":

    wandb.agent(sweep_id, function=train, count=5)
