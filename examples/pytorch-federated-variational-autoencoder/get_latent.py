from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F


from utils_mnist import (
    test,
    visualize_gen_image,
    visualize_gmm_latent_representation,
    non_iid_train_iid_test_6789,
    subset_alignment_dataloader,
    train_align,
    eval_reconstrution,
)
from utils_mnist import VAE
import os
import numpy as np
import matplotlib.pyplot as plt
from utils_mnist import VAE

NUM_CLIENTS = 2
NUM_CLASSES = 4
samples_per_class = 200
alignment_dataloader = subset_alignment_dataloader(
    samples_per_class=samples_per_class,
    batch_size=samples_per_class * NUM_CLASSES,
)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb


def vae_loss(recon_img, img, mu, logvar, beta=1):
    # Reconstruction loss using binary cross-entropy
    condition = (recon_img >= 0.0) & (recon_img <= 1.0)
    # assert torch.all(condition), "Values should be between 0 and 1"
    # if not torch.all(condition):
    #     ValueError("Values should be between 0 and 1")
    #     recon_img = torch.clamp(recon_img, 0.0, 1.0)
    recon_loss = F.binary_cross_entropy(
        recon_img, img.view(-1, img.shape[2] * img.shape[3]), reduction="sum"
    )
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total VAE loss
    total_loss = recon_loss + kld_loss * beta
    return total_loss


def visualise_latent(ref_model, ep):
    ref_model.eval()
    with torch.no_grad():
        test_latents = []
        test_labels = []  # Store corresponding labels
        for x, labels in alignment_dataloader:  # Retrieve labels from test_loader
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)
            recon_images, mu, logvar = ref_model(x)
            test_latents.append(mu.cpu().numpy())
            test_labels.append(labels.cpu().numpy())  # Store labels

    # Visualize latent representations in 2D
    test_latents = np.concatenate(test_latents, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)  # Concatenate all labels
    from sklearn.decomposition import PCA

    # Assuming test_latents is a numpy array with shape (num_samples, 16)
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    test_latents_pca = pca.fit_transform(test_latents)
    plt.figure(figsize=(8, 6))
    for label in np.unique(test_labels):
        indices = np.where(test_labels == label)
        # print(indices)
        # print(indices[0])  # Use [0] to access indices
        plt.scatter(
            test_latents[indices, 0],
            test_latents[indices, 1],
            label=label,
            alpha=1,
        )
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Latent Space Visualization of Test Set with Class Highlight")
    plt.legend()
    plt.savefig("latent_space_visualization.png")  # Save the figure as PNG

    wandb.log(
        {"Latent Space Visualization": wandb.Image("latent_space_visualization.png")},
        step=ep + 1,
    )
    plt.close()


def main():
    run = wandb.init(
        entity="mak",
        group="get_posterior",
        reinit=True,
    )
    print(f"configuraion:{wandb.config}")
    ref_model = VAE(z_dim=2).to(DEVICE)
    opt_ref = torch.optim.Adam(ref_model.parameters(), lr=wandb.config["lr"])
    for ep in range(5000):
        for images, _ in alignment_dataloader:
            images = images.to(DEVICE)
            opt_ref.zero_grad()
            recon_images, mu, logvar = ref_model(images)
            vae_loss1 = vae_loss(recon_images, images, mu, logvar, wandb.config["beta"])
            vae_loss1.backward()
            opt_ref.step()

        # if ep % 100 == 0:
        #     print(f"Epoch {ep}, Loss {vae_loss1.item()}")
        wandb.log({"Loss": vae_loss1.item()}, step=ep + 1)

        # print("--------------------------------------------------")
        visualise_latent(ref_model, ep)


if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {"name": "vae_loss1", "goal": "minimize"},
        "parameters": {
            "beta": {"values": [8, 5, 10, 2]},
            "lr": {
                "values": [
                    1e-3,
                    1e-4,
                ]
            },
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project="get_posterior")

    wandb.agent(sweep_id, function=main, count=5)
