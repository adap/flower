import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from collections import OrderedDict
import flwr as fl
from flwr.common import parameters_to_ndarrays
from torch.nn.parameter import Parameter
from typing import List, Tuple
from sklearn.mixture import GaussianMixture
import matplotlib
import plotly.graph_objects as go

matplotlib.use("Agg")


class Net(nn.Module):
    def __init__(self, h_dim=64, z_dim=10) -> None:
        super(Net, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, h_dim),
            nn.ReLU(),
            # nn.Linear(h_dim, h_dim // 2),
            # nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            # nn.Linear(h_dim // 2, h_dim),
            # nn.ReLU(),
            nn.Linear(h_dim, 28 * 28),
            nn.Sigmoid(),  # Use Sigmoid activation for MNIST (pixel values between 0 and 1)
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        x = x.view(x.size(0), -1)  # Flatten input for fully connected layers
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x_recon

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


class VAE(nn.Module):
    def __init__(
        self, x_dim=784, h_dim1=512, h_dim2=256, h_dim3=32, z_dim=2, encoder_only=False
    ):
        super(VAE, self).__init__()
        self.encoder_only = encoder_only
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc41 = nn.Linear(h_dim3, z_dim)
        self.fc42 = nn.Linear(h_dim3, z_dim)
        # decoder part
        self.fc5 = nn.Linear(z_dim, h_dim3)
        self.fc6 = nn.Linear(h_dim3, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc41(h), self.fc42(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return torch.sigmoid(self.fc8(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        if self.encoder_only:
            output = z
        else:
            output = self.decoder(z)

        return output, mu, log_var


class VAE_CNN(nn.Module):
    def __init__(self, z_dim=2, encoder_only=False):
        super(VAE_CNN, self).__init__()
        self.encoder_only = encoder_only

        # Encoder layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # Assuming input size 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 7 * 7, z_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, z_dim)

        # Decoder layers
        self.fc4 = nn.Linear(z_dim, 128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 7 * 7)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        x = F.relu(self.fc4(z))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        if self.encoder_only:
            output = z
        else:
            output = self.decoder(z)
        return output, mu, log_var


def alignment_dataloader(
    samples_per_class=100, batch_size=8, shuffle=False, only_data=False
):
    # Load the MNIST test dataset
    mnist_test = MNIST(
        root="./mnist_data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Create an alignment dataset with 20 samples for each class
    alignment_datasets = []

    for class_label in range(10):
        class_indices = [
            i for i, (img, label) in enumerate(mnist_test) if label == class_label
        ]
        # print(f"{class_label}:{len(class_indices)}")
        selected_indices = class_indices[:samples_per_class]
        alignment_dataset = Subset(mnist_test, selected_indices)
        alignment_datasets.append(alignment_dataset)

    # Concatenate the alignment datasets into one
    alignment_dataset = ConcatDataset(alignment_datasets)
    if only_data:
        return alignment_dataset

    # Create a DataLoader for the alignment dataset
    alignment_loader = DataLoader(
        alignment_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return alignment_loader


def alignment_dataloader_wo_9(samples_per_class=100, batch_size=8, shuffle=False):
    # Load the MNIST test dataset
    mnist_test = MNIST(
        root="./mnist_data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Create an alignment dataset with 20 samples for each class
    alignment_datasets = []
    # excluding 9
    for class_label in range(9):
        class_indices = [
            i for i, (img, label) in enumerate(mnist_test) if label == class_label
        ]
        selected_indices = class_indices[:samples_per_class]
        alignment_dataset = Subset(mnist_test, selected_indices)
        alignment_datasets.append(alignment_dataset)
    # Concatenate the alignment datasets into one
    alignment_dataset = ConcatDataset(alignment_datasets)

    # Create a DataLoader for the alignment dataset
    alignment_loader = DataLoader(
        alignment_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return alignment_loader


def subset_alignment_dataloader(samples_per_class=100, batch_size=8, shuffle=True):
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=transforms.ToTensor()
    )
    partitions_idx = non_iid_train_iid_test_6789(alignment=True)
    torch.manual_seed(6789)
    alignment_datasets = []
    for partition_idx in partitions_idx:
        selected_points = partition_idx[:samples_per_class]
        alignment_dataset = Subset(test_dataset, selected_points)
        alignment_datasets.append(alignment_dataset)

    # Concatenate the alignment datasets into one
    alignment_dataset = ConcatDataset(alignment_datasets)

    # Create a DataLoader for the alignment dataset
    alignment_loader = DataLoader(
        alignment_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return alignment_loader


def load_data_mnist(normalise=False, batch_size=64):
    """Load MNIST (training and test set)."""
    if normalise:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        transform = transforms.ToTensor()

    trainset = MNIST(
        root="./mnist_data/", train=True, download=True, transform=transform
    )
    testset = MNIST(
        root="./mnist_data/", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader


def non_iid_train_iid_test():
    # Load the MNIST training dataset
    train_dataset = MNIST(
        root="./mnist_data/", train=True, download=True, transform=transforms.ToTensor()
    )

    # Define class pairs for each partition
    class_partitions = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # Create a list to store datasets for each partition
    partition_datasets_train = []

    # Iterate over class pairs and create a dataset for each partition
    for class_pair in class_partitions:
        class_filter = lambda label: label in class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(train_dataset) if class_filter(label)
        ]

        # Use Subset to create a dataset with filtered indices
        partition_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)
        partition_datasets_train.append(partition_dataset)

        # Load the MNIST test dataset
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Specify the size of each partition
    partition_sizes = [len(test_dataset) // 5] * 4 + [
        len(test_dataset) - (len(test_dataset) // 5) * 4
    ]

    # Use random_split to create 5 datasets with random samples
    partition_datasets_test = torch.utils.data.random_split(
        test_dataset, partition_sizes
    )
    return partition_datasets_train, partition_datasets_test


def non_iid_wo9_train_iid_test():
    # Load the MNIST training dataset
    train_dataset = MNIST(
        root="./mnist_data/", train=True, download=True, transform=transforms.ToTensor()
    )

    # Define class pairs for each partition
    class_partitions = [(0, 1), (2, 3), (4, 5), (6, 7), (8,)]

    # Create a list to store datasets for each partition
    partition_datasets_train = []

    # Iterate over class pairs and create a dataset for each partition
    for class_pair in class_partitions:
        class_filter = lambda label: label in class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(train_dataset) if class_filter(label)
        ]

        # Use Subset to create a dataset with filtered indices
        partition_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)
        partition_datasets_train.append(partition_dataset)

        # Load the MNIST test dataset
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Specify the size of each partition
    partition_sizes = [len(test_dataset) // 5] * 4 + [
        len(test_dataset) - (len(test_dataset) // 5) * 4
    ]

    # Use random_split to create 5 datasets with random samples
    partition_datasets_test = torch.utils.data.random_split(
        test_dataset, partition_sizes
    )
    return partition_datasets_train, partition_datasets_test


def non_iid_train_iid_test_6789(seed=6789, alignment=False):
    # Load the MNIST training dataset
    torch.manual_seed(seed)

    train_dataset = MNIST(
        root="./mnist_data/", train=True, download=True, transform=transforms.ToTensor()
    )

    # Define class pairs for each partition
    class_partitions = [(6, 7), (8, 9)]

    # Create a list to store datasets for each partition
    partition_datasets_train = []

    # Iterate over class pairs and create a dataset for each partition
    for class_pair in class_partitions:
        class_filter = lambda label: label in class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(train_dataset) if class_filter(label)
        ]

        # Use Subset to create a dataset with filtered indices
        partition_dataset = Subset(train_dataset, filtered_indices)
        partition_datasets_train.append(partition_dataset)

        # Load the MNIST test dataset
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=transforms.ToTensor()
    )
    class_partitions_test = [6, 7, 8, 9]

    partition_datasets_test = []
    partition_datasets_alignment = []

    # Iterate over class pairs and create a dataset for each partition
    for class_pair in class_partitions_test:
        class_filter = class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(test_dataset) if class_filter == label
        ]

        # Use Subset to create a dataset with filtered indices
        partition_datasets_test.append(Subset(test_dataset, filtered_indices[500:]))
        partition_datasets_alignment.append(filtered_indices[:500])

    if alignment:
        return partition_datasets_alignment
    combined_testset = [ConcatDataset(partition_datasets_test)] * len(
        partition_datasets_train
    )
    return (
        partition_datasets_train,
        combined_testset,
    )


def iid_train_iid_test():
    # Load the MNIST training dataset
    train_dataset = MNIST(
        root="./mnist_data/", train=True, download=True, transform=transforms.ToTensor()
    )

    # Specify the size of each partition
    partition_sizes_train = [len(train_dataset) // 5] * 4 + [
        len(train_dataset) - (len(train_dataset) // 5) * 4
    ]

    # Use random_split to create 5 datasets with random samples
    partition_datasets_train = torch.utils.data.random_split(
        train_dataset, partition_sizes_train
    )
    # Load the MNIST test dataset
    test_dataset = MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Specify the size of each partition
    partition_sizes_test = [len(test_dataset) // 5] * 4 + [
        len(test_dataset) - (len(test_dataset) // 5) * 4
    ]

    # Use random_split to create 5 datasets with random samples
    partition_datasets_test = torch.utils.data.random_split(
        test_dataset, partition_sizes_test
    )
    return partition_datasets_train, partition_datasets_test


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(
    net,
    trainloader,
    optimizer,
    config,
    epochs,
    device,
    num_classes=None,
    if_return=False,
):
    """Train the network on the training set."""
    net.train()
    for _ in range(epochs):
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)

            bce_loss_per_pixel = F.binary_cross_entropy(
                recon_images,
                images.view(-1, images.shape[2] * images.shape[3]),
                reduction="none",
            )
            # Sum along dimension 1 (sum over pixels for each image)
            bce_loss_sum_per_image = torch.sum(
                bce_loss_per_pixel, dim=1
            )  # Shape: (batch_size,)

            # Take the mean along dimension 0 (mean over images in the batch)
            recon_loss = torch.mean(bce_loss_sum_per_image)  # Shape: scalar

            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kld_loss = torch.mean(kld_loss)
            loss = recon_loss + kld_loss * config["beta"]
            loss.backward()
            optimizer.step()
    return loss.item()
    if if_return:
        return net


def train_prox(
    net,
    trainloader,
    optim,
    config,
    epochs,
    device,
    num_classes,
):
    criterion = None  # loss in functional form
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net, vae_term, prox_term = _train_one_epoch(
            net,
            global_params,
            trainloader,
            device,
            criterion,
            optim,
            config.get("proximal_mu", 1),
        )
    return vae_term, prox_term


def compute_ref_stats(alignment_loader, config, device, use_PCA=True):
    ref_model = VAE(z_dim=config["latent_dim"]).to(device)
    opt_ref = torch.optim.Adam(ref_model.parameters(), lr=1e-3)
    for ep in range(config["prior_steps"]):
        for images, labels in alignment_loader:
            images = images.to(device)
            # labels = labels.to(device)
            opt_ref.zero_grad()
            recon_images, mu, logvar = ref_model(images)
            total_loss = vae_loss(recon_images, images, mu, logvar, 1)
            total_loss.backward()
            opt_ref.step()
        if ep % 100 == 0:
            print(f"Epoch {ep}, Loss {total_loss.item()}")

            print(f"--------------------------------------------------")
    ref_model.eval()
    test_latents = []
    test_labels = []
    with torch.no_grad():
        for images, labels in alignment_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, ref_mu, ref_logvar = ref_model(images)
            test_latents.append(ref_mu.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
        test_latents = np.concatenate(test_latents, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        if use_PCA:

            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            reduced_latents = pca.fit_transform(test_latents)
        # Create traces for each class
        traces = []
        for label in np.unique(test_labels):
            indices = np.where(test_labels == label)
            trace = go.Scatter(
                x=reduced_latents[indices, 0].flatten(),
                y=reduced_latents[indices, 1].flatten(),
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
    return (ref_mu, ref_logvar), fig


def _train_one_epoch(
    net,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        recon_images, mu, logvar = net(images)

        bce_loss_per_pixel = F.binary_cross_entropy(
            recon_images,
            images.view(-1, images.shape[2] * images.shape[3]),
            reduction="none",
        )
        # Sum along dimension 1 (sum over pixels for each image)
        bce_loss_sum_per_image = torch.sum(
            bce_loss_per_pixel, dim=1
        )  # Shape: (batch_size,)

        # Take the mean along dimension 0 (mean over images in the batch)
        recon_loss = torch.mean(bce_loss_sum_per_image)  # Shape: scalar
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss + (proximal_mu / 2) * proximal_term

        vae_term = recon_loss + kld_loss
        prox_term = (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net, vae_term, prox_term


def vae_loss(recon_img, img, mu, logvar, beta=1.0, cnn=False):
    # Reconstruction loss using binary cross-entropy
    # condition = (recon_img >= 0) & (recon_img <= 1)
    # import pdb; pdb.set_trace()
    # assert torch.all(condition), "Values should be between 0 and 1"

    if cnn:
        # Reconstruction loss using Mean Squared Error (MSE)
        mse_loss_per_pixel = F.mse_loss(recon_img, img, reduction="none")
        # Sum along dimensions except the batch dimension
        mse_loss_sum_per_image = torch.sum(
            mse_loss_per_pixel, dim=(1, 2, 3)
        )  # Shape: (batch_size,)

        # Take the mean along dimension 0 (mean over images in the batch)
        recon_loss = torch.mean(mse_loss_sum_per_image)  # Shape: scalar
    else:
        bce_loss_per_pixel = F.binary_cross_entropy(
            recon_img, img.view(-1, img.shape[2] * img.shape[3]), reduction="none"
        )
        # Sum along dimension 1 (sum over pixels for each image)
        bce_loss_sum_per_image = torch.sum(
            bce_loss_per_pixel, dim=1
        )  # Shape: (batch_size,)

        # Take the mean along dimension 0 (mean over images in the batch)
        recon_loss = torch.mean(bce_loss_sum_per_image)  # Shape: scalar

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # Take the mean along dimension 0 (mean over images in the batch)
    kld_loss = torch.mean(kld_loss)
    # Total VAE loss
    total_loss = recon_loss + kld_loss * beta

    return total_loss


def vae_rec_loss(recon_img, img, cnn=False):
    # Reconstruction loss using binary cross-entropy
    # condition = (recon_img >= 0) & (recon_img <= 1)
    # assert torch.all(condition), "Values should be between 0 and 1"
    if cnn:
        # Reconstruction loss using Mean Squared Error (MSE)
        mse_loss_per_pixel = F.mse_loss(recon_img, img, reduction="none")
        # Sum along dimensions except the batch dimension
        mse_loss_sum_per_image = torch.sum(
            mse_loss_per_pixel, dim=(1, 2, 3)
        )  # Shape: (batch_size,)
        return mse_loss_sum_per_image
    else:
        bce_loss_per_pixel = F.binary_cross_entropy(
            recon_img, img.view(-1, img.shape[2] * img.shape[3]), reduction="none"
        )
        # Sum along dimension 1 (sum over pixels for each image)
        bce_loss_sum_per_image = torch.sum(
            bce_loss_per_pixel, dim=1
        )  # Shape: (batch_size,)
        return bce_loss_sum_per_image


def train_align(
    net,
    trainloader,
    align_loader,
    optimizer,
    config,
    epochs,
    device,
    num_classes=None,
):
    """Train the network on the training set."""
    net.train()
    temp_gen_model = VAE(z_dim=config["latent_dim"], encoder_only=True).to(device)
    gen_weights = parameters_to_ndarrays(config["gen_params"])
    params_dict = zip(temp_gen_model.state_dict().keys(), gen_weights)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    temp_gen_model.load_state_dict(state_dict, strict=True)

    temp_gen_model.eval()
    # fixed_gen_stats = config["gen_params"]
    # print(f"fixed mu: {fixed_gen_stats[0]}")
    print(f"gen_weights: {gen_weights[7]}")

    lambda_reg = config["lambda_reg"]

    lambda_align = config["lambda_align"]
    beta = config["beta"]

    for _ in range(epochs):
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            vae_loss1 = vae_loss(recon_images, images, mu, logvar, beta)
            z_g, mu_g, logvar_g = temp_gen_model(images)
            vae_loss2 = vae_loss(net.decoder(z_g), images, mu_g, logvar_g, beta)
            loss = vae_loss1
            loss += lambda_reg * vae_loss2
            for align_epoch in range(100):
                accumulate_align_loss = 0
                for align_img, _ in align_loader:
                    align_img = align_img.to(device)
                    _, mu_g, log_var_g = temp_gen_model(align_img)
                    _, mu, log_var = net(align_img)
                    # mu_g, log_var_g = fixed_gen_stats

                    loss_align = 0.5 * (log_var_g - log_var - 1) + (
                        log_var.exp() + (mu - mu_g).pow(2)
                    ) / (2 * log_var_g.exp())

                loss_align_reduced = torch.mean(loss_align.sum(dim=1))
                accumulate_align_loss += loss_align_reduced
                print(f"lambda_align: {lambda_align }")
                print(f"loss_align_term: {lambda_align * loss_align_reduced}")
            avg_accumulate_align_loss = accumulate_align_loss / (align_epoch + 1)
            loss += lambda_align * avg_accumulate_align_loss
            loss.backward()
            optimizer.step()

    return (
        vae_loss1.item(),
        # 0,
        lambda_reg * vae_loss2.item(),
        lambda_align * avg_accumulate_align_loss.item(),
    )


def train_align_dec_frozen(
    net,
    trainloader,
    align_loader,
    optimizer,
    config,
    epochs,
    device,
    num_classes=None,
):
    """Train the network on the training set."""
    net.train()
    temp_gen_model = VAE(z_dim=config["latent_dim"], encoder_only=True).to(device)
    gen_weights = parameters_to_ndarrays(config["gen_params"])
    params_dict = zip(temp_gen_model.state_dict().keys(), gen_weights)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    temp_gen_model.load_state_dict(state_dict, strict=True)

    temp_gen_model.eval()
    # fixed_gen_stats = config["gen_params"]
    # print(f"fixed mu: {fixed_gen_stats[0]}")
    print(f"gen_weights: {gen_weights[7]}")

    lambda_reg = config["lambda_reg"]

    lambda_align = config["lambda_align"]
    lambda_latent_diff = config["lambda_latent_diff"]
    # lambda_reg_dec = config["lambda_reg_dec"]
    beta = config["beta"]
    latent_diff_loss = nn.MSELoss(reduction="none")
    opt1 = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Freeze decoder parameters
    encoder_params = (
        list(net.fc1.parameters())
        + list(net.fc2.parameters())
        + list(net.fc3.parameters())
        + list(net.fc41.parameters())
        + list(net.fc42.parameters())
    )
    decorder_params = (
        list(net.fc5.parameters())
        + list(net.fc6.parameters())
        + list(net.fc7.parameters())
        + list(net.fc8.parameters())
    )
    opt2 = torch.optim.Adam(encoder_params, lr=1e-3)
    opt3 = torch.optim.Adam(decorder_params, lr=1e-3)
    for _ in range(epochs):
        # for images, _ in trainloader:
        for idx, (align_img, _) in enumerate(trainloader):
            opt1.zero_grad()
            align_img = align_img.to(device)
            z_g, mu_g, log_var_g = temp_gen_model(align_img)
            recon_align_img, mu, log_var = net(align_img)
            vae_loss1 = vae_loss(recon_align_img, align_img, mu, log_var, beta)
            loss_ref = vae_loss1
            vae_loss_g = vae_loss(net.decoder(z_g), align_img, mu_g, log_var_g, beta)
            loss_ref += vae_loss_g * lambda_reg

            loss_align = 0.5 * (log_var_g - log_var - 1) + (
                log_var.exp() + (mu - mu_g).pow(2)
            ) / (2 * log_var_g.exp())

        loss_align_reduced = torch.mean(loss_align.sum(dim=1))
        print(f"align_step: {idx }")
        print(f"loss_align_term: {lambda_align * loss_align_reduced}")

        loss_ref += lambda_align * loss_align_reduced
        loss_ref.backward()
        opt1.step()

        # opt2.zero_grad()
        # images = images.to(device)
        # recon_images, mu, logvar = net(images)
        # z = net.sampling(mu, logvar)
        # z_g, _, _ = temp_gen_model(images)

        # vae_loss2 = latent_diff_loss(z, z_g).sum(dim=1).mean()
        # loss_local = vae_loss(recon_images, images, mu, logvar, beta)
        # loss_local += lambda_latent_diff * vae_loss2

        # loss_local.backward()
        # opt2.step()

        # opt3.zero_grad()
        # recon_images, mu, logvar = net(images)
        # loss_local2 = (
        #     vae_loss(recon_images, images, mu, logvar, beta) * lambda_reg_dec
        # )
        # loss_local2.backward()
        # opt3.step()
    # TODO:update for local data
    return (
        0,  # loss_local.item(),
        loss_ref.item(),
        lambda_align * loss_align_reduced.item(),
        0,  # lambda_latent_diff * vae_loss2.item(),
    )


def train_alternate_frozen(
    net,
    trainloader,
    align_loader,
    config,
    epochs,
    device,
) -> Tuple[float, float, float, float]:
    """Train the network on the training set."""
    net.train()
    if config["cnn"]:
        temp_gen_model = VAE_CNN(z_dim=config["latent_dim"], encoder_only=True).to(
            device
        )
    else:
        temp_gen_model = VAE(z_dim=config["latent_dim"], encoder_only=True).to(device)
    gen_weights = parameters_to_ndarrays(config["gen_params"])
    params_dict = zip(temp_gen_model.state_dict().keys(), gen_weights)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    temp_gen_model.load_state_dict(state_dict, strict=True)

    temp_gen_model.eval()

    print(f"gen_weights: {gen_weights[7]}")

    lambda_reg = config["lambda_reg"]

    lambda_align = config["lambda_align"]
    lambda_align2 = config["lambda_align2"]
    lambda_latent_diff = config["lambda_latent_diff"]
    lambda_reg_dec = config["lambda_reg_dec"]
    beta = config["beta"]
    latent_diff_loss = nn.MSELoss(reduction="none")
    opt1 = torch.optim.Adam(net.parameters(), lr=1e-3)
    if config["cnn"]:
        encoder_params = (
            list(net.conv1.parameters())
            + list(net.conv2.parameters())
            + list(net.conv3.parameters())
            + list(net.fc_mu.parameters())
            + list(net.fc_logvar.parameters())
        )
        decorder_params = (
            list(net.fc4.parameters())
            + list(net.deconv1.parameters())
            + list(net.deconv2.parameters())
            + list(net.deconv3.parameters())
        )
    else:
        encoder_params = (
            list(net.fc1.parameters())
            + list(net.fc2.parameters())
            + list(net.fc3.parameters())
            + list(net.fc41.parameters())
            + list(net.fc42.parameters())
        )
        decorder_params = (
            list(net.fc5.parameters())
            + list(net.fc6.parameters())
            + list(net.fc7.parameters())
            + list(net.fc8.parameters())
        )
    opt2 = torch.optim.Adam(encoder_params, lr=1e-3)
    opt3 = torch.optim.Adam(decorder_params, lr=1e-3)
    for _ in range(epochs):
        for images, _ in trainloader:
            for idx, (align_img, _) in enumerate(align_loader):
                opt1.zero_grad()
                align_img = align_img.to(device)
                z_g, mu_g, log_var_g = temp_gen_model(align_img)
                recon_align_img, mu, log_var = net(align_img)
                vae_loss1 = vae_loss(
                    recon_align_img, align_img, mu, log_var, beta, cnn=True
                )
                loss_ref = vae_loss1
                vae_loss_g = vae_loss(
                    net.decoder(z_g), align_img, mu_g, log_var_g, beta, cnn=True
                )
                loss_ref += vae_loss_g * lambda_reg
                if lambda_align > 0:
                    loss_align = 0.5 * (log_var_g - log_var - 1) + (
                        log_var.exp() + (mu - mu_g).pow(2)
                    ) / (2 * log_var_g.exp())
                else:
                    loss_align = torch.zeros_like(log_var_g)

            loss_align_reduced = torch.mean(loss_align.sum(dim=1))
            print(f"align_step: {idx }")
            print(f"loss_align_term: {lambda_align * loss_align_reduced}")

            loss_ref += lambda_align * loss_align_reduced
            loss_ref.backward(retain_graph=False)
            opt1.step()

            opt2.zero_grad()
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            z = net.sampling(mu, logvar)
            z_g2, mu_g2, log_var_g2 = temp_gen_model(images)

            loss_local = vae_loss(recon_images, images, mu, logvar, beta, cnn=True)
            vae_loss2 = latent_diff_loss(z, z_g2).sum(dim=1).mean()
            loss_local += lambda_latent_diff * vae_loss2
            if lambda_align2 > 0:
                loss_align = 0.5 * (log_var_g2 - logvar - 1) + (
                    logvar.exp() + (mu - mu_g2).pow(2)
                ) / (2 * log_var_g2.exp())
                loss_local += lambda_align2 * loss_align.sum(dim=1).mean()

            loss_local.backward(retain_graph=False)
            opt2.step()
            if lambda_reg_dec > 0:
                print("dec update")
                opt3.zero_grad()
                recon_images, mu, logvar = net(images)
                loss_local2 = (
                    vae_loss(recon_images, images, mu, logvar, beta, cnn=True)
                    * lambda_reg_dec
                )
                loss_local2.backward()
                opt3.step()

    # TODO:update for local data
    return (
        loss_local.item(),
        loss_ref.item(),
        lambda_align * loss_align_reduced.item(),
        lambda_latent_diff * vae_loss2.item(),
    )


def train_single_loader(
    net,
    trainloader,
    align_loader,
    optimizer,
    config,
    epochs,
    device,
    num_classes=None,
):
    """Train the network on the training set."""
    net.train()
    temp_gen_model = VAE(z_dim=config["latent_dim"], encoder_only=True).to(device)
    gen_weights = parameters_to_ndarrays(config["gen_params"])
    params_dict = zip(temp_gen_model.state_dict().keys(), gen_weights)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    temp_gen_model.load_state_dict(state_dict, strict=True)

    temp_gen_model.eval()
    # fixed_gen_stats = config["gen_params"]
    # print(f"fixed mu: {fixed_gen_stats[0]}")
    print(f"gen_weights: {gen_weights[7]}")

    lambda_reg = config["lambda_reg"]

    lambda_align = config["lambda_align"]
    beta = config["beta"]
    opt1 = torch.optim.Adam(net.parameters(), lr=1e-3)

    # # Freeze decoder parameters
    # encoder_params = (
    #     list(net.fc1.parameters())
    #     + list(net.fc2.parameters())
    #     + list(net.fc3.parameters())
    #     + list(net.fc41.parameters())
    #     + list(net.fc42.parameters())
    # )
    # decorder_params = (
    #     list(net.fc5.parameters())
    #     + list(net.fc6.parameters())
    #     + list(net.fc7.parameters())
    #     + list(net.fc8.parameters())
    # )
    # opt2 = torch.optim.Adam(encoder_params, lr=1e-3)
    # opt3 = torch.optim.Adam(decorder_params, lr=1e-3)
    for _ in range(epochs):
        # for images, _ in trainloader:
        for idx, (align_img, _) in enumerate(trainloader):
            opt1.zero_grad()
            align_img = align_img.to(device)
            z_g, mu_g, log_var_g = temp_gen_model(align_img)
            recon_align_img, mu, log_var = net(align_img)
            vae_loss1 = vae_loss(recon_align_img, align_img, mu, log_var, beta)
            loss_ref = vae_loss1
            vae_loss_g = vae_loss(net.decoder(z_g), align_img, mu_g, log_var_g, beta)
            loss_ref += vae_loss_g * lambda_reg

            loss_align = 0.5 * (log_var_g - log_var - 1) + (
                log_var.exp() + (mu - mu_g).pow(2)
            ) / (2 * log_var_g.exp())

        loss_align_reduced = torch.mean(loss_align.sum(dim=1))
        print(f"align_step: {idx }")
        print(f"loss_align_term: {lambda_align * loss_align_reduced}")

        loss_ref += lambda_align * loss_align_reduced
        loss_ref.backward()
        opt1.step()

        # opt2.zero_grad()
        # images = images.to(device)
        # recon_images, mu, logvar = net(images)
        # z = net.sampling(mu, logvar)
        # z_g, _, _ = temp_gen_model(images)

        # vae_loss2 = latent_diff_loss(z, z_g).sum(dim=1).mean()
        # loss_local = vae_loss(recon_images, images, mu, logvar, beta)
        # loss_local += lambda_latent_diff * vae_loss2

        # loss_local.backward()
        # opt2.step()

        # opt3.zero_grad()
        # recon_images, mu, logvar = net(images)
        # loss_local2 = (
        #     vae_loss(recon_images, images, mu, logvar, beta) * lambda_reg_dec
        # )
        # loss_local2.backward()
        # opt3.step()
    # TODO:update for local data
    return (
        0,  # loss_local.item(),
        loss_ref.item(),
        lambda_align * loss_align_reduced.item(),
        0,  # lambda_latent_diff * vae_loss2.item(),
    )


def train_align_prox(
    net,
    trainloader,
    align_loader,
    optimizer,
    config,
    epochs,
    device,
    num_classes=None,
):
    """Train the network on the training set."""
    net.train()
    # temp_gen_model = VAE(z_dim=config["latent_dim"], encoder_only=True).to(device)
    # gen_weights = parameters_to_ndarrays(config["gen_params"])
    # params_dict = zip(temp_gen_model.state_dict().keys(), gen_weights)
    # state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    # temp_gen_model.load_state_dict(state_dict, strict=True)
    # temp_gen_model.eval()

    beta = config["beta"]
    mu_g = config["mu_g"]
    log_var_g = config["log_var_g"]
    lambda_align = config["lambda_align"]

    for _ in range(epochs):
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            vae_loss1 = vae_loss(recon_images, images, mu, logvar, beta)
            # z_g, mu_g, logvar_g = temp_gen_model(images)
            # vae_loss2 = vae_loss(net.decoder(z_g), images, mu_g, logvar_g, beta)
            loss = vae_loss1
            accumulate_align_loss = 0
            for align_img, _ in align_loader:
                align_img = align_img.to(device)

                _, mu, log_var = net(align_img)

                loss_align = 0.5 * (log_var_g - log_var - 1) + (
                    log_var.exp() + (mu - mu_g).pow(2)
                ) / (2 * log_var_g.exp())
                accumulate_align_loss += torch.mean(loss_align.sum(dim=1))
            loss_align_reduced = accumulate_align_loss
            loss += lambda_align * loss_align_reduced
            print(f"align term: {lambda_align * loss_align_reduced.item() }")
            loss.backward()
            optimizer.step()

    return (
        vae_loss1.item(),
        lambda_align * loss_align_reduced.item(),
    )


def test(net, testloader, device, kl_term=0, cnn=False):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images = data[0].to(device)
            recon_images, mu, logvar = net(images)

            if cnn:
                # Reconstruction loss using Mean Squared Error (MSE)
                mse_loss_per_pixel = F.mse_loss(recon_images, images, reduction="none")
                # Sum along dimensions except the batch dimension
                mse_loss_sum_per_image = torch.sum(
                    mse_loss_per_pixel, dim=(1, 2, 3)
                )  # Shape: (batch_size,)

                # Take the mean along dimension 0 (mean over images in the batch)
                recon_loss = torch.mean(mse_loss_sum_per_image)  # Shape: scalar
            else:
                bce_loss_per_pixel = F.binary_cross_entropy(
                    recon_images,
                    images.view(-1, images.shape[2] * images.shape[3]),
                    reduction="none",
                )
                # Sum along dimension 1 (sum over pixels for each image)
                bce_loss_sum_per_image = torch.sum(
                    bce_loss_per_pixel, dim=1
                )  # Shape: (batch_size,)

                # Take the mean along dimension 0 (mean over images in the batch)
                recon_loss = torch.mean(bce_loss_sum_per_image)  # Shape: scalar
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss * kl_term
            total += len(images)
    # TODO: accu=-1*loss
    # return (loss.item() / total), -1 * (loss.item() / total)
    # batch_loss = loss.item()
    return (loss.item() / (idx + 1)), -1 * (loss.item() / (idx + 1))


def eval_reconstrution(net, testloader, device, cnn=False):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images = data[0].to(device)
            recon_images, mu, logvar = net(images)
            if cnn:
                # Reconstruction loss using Mean Squared Error (MSE)
                mse_loss_per_pixel = F.mse_loss(recon_images, images, reduction="none")
                # Sum along dimensions except the batch dimension
                mse_loss_sum_per_image = torch.sum(
                    mse_loss_per_pixel, dim=(1, 2, 3)
                )  # Shape: (batch_size,)

                # Take the mean along dimension 0 (mean over images in the batch)
                recon_loss = torch.mean(mse_loss_sum_per_image)  # Shape: scalar
            else:
                bce_loss_per_pixel = F.binary_cross_entropy(
                    recon_images,
                    images.view(-1, images.shape[2] * images.shape[3]),
                    reduction="none",
                )
                # Sum along dimension 1 (sum over pixels for each image)
                bce_loss_sum_per_image = torch.sum(
                    bce_loss_per_pixel, dim=1
                )  # Shape: (batch_size,)

                # Take the mean along dimension 0 (mean over images in the batch)
                recon_loss = torch.mean(bce_loss_sum_per_image)  # Shape: scalar

            loss += recon_loss
            # total += len(images)
    return loss.item() / (idx + 1)


def visualize_gen_image(net, testloader, device, rnd=None, folder=None):
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
        return (
            f"{folder}/true_img_at_{rnd}.png",
            f"{folder}/test_generated_at_{rnd}.png",
        )


def visualize_latent_representation(
    model, test_loader, device, rnd=None, folder=None, use_PCA=False
):
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
    reduced_latents = all_latents
    if use_PCA:
        # Apply PCA using PyTorch
        cov_matrix = torch.tensor(np.cov(all_latents.T), dtype=torch.float32)
        _, _, V = torch.svd_lowrank(cov_matrix, q=2)

        # Project data onto the first two principal components
        reduced_latents = torch.mm(torch.tensor(all_latents, dtype=torch.float32), V)

        # Convert to numpy array
        reduced_latents = reduced_latents.numpy()
    plt.figure(figsize=(10, 8))
    # Visualize latent representation
    scatter = plt.scatter(
        reduced_latents[:, 0], reduced_latents[:, 1], c=all_labels, cmap="tab10"
    )
    plt.colorbar(scatter, label="Digit Label")
    plt.title("Latent Representation Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig(f"{folder}/latent_rep_at_{rnd}.png")


def visualize_gmm_latent_representation(
    model, test_loader, device, rnd=None, folder=None, use_PCA=False, num_class=10
):
    model.eval()
    all_latents = []
    all_labels = []
    all_means = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            z, mu, _ = model(data)
            all_latents.append(z.cpu().numpy())
            all_means.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    reduced_latents = all_latents
    if use_PCA:
        # Apply PCA using PyTorch
        cov_matrix = torch.tensor(np.cov(all_latents.T), dtype=torch.float32)
        _, _, V = torch.svd_lowrank(cov_matrix, q=2)

        # Project data onto the first two principal components
        reduced_latents = torch.mm(torch.tensor(all_latents, dtype=torch.float32), V)

        # Convert to numpy array
        reduced_latents = reduced_latents.numpy()
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter2 = ax.scatter(
        reduced_latents[:, 0],
        reduced_latents[:, 1],
        c=all_labels,
        cmap="Set1",
        label="Labels",
        zorder=2,
    )
    ax.set_title("Latent Representation with True Labels")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    ax.grid()
    # Create a colorbar for the scatter plots
    # cbar1 = plt.colorbar(scatter1, ax=axs[0], label="GMM Predictions")
    cbar2 = plt.colorbar(scatter2, ax=ax, label="True Labels")

    # Set colorbar ticks and labels based on unique label values
    unique_labels = np.unique(all_labels)

    cbar2.set_ticks(unique_labels)
    cbar2.set_ticklabels(unique_labels)

    # Adjust layout for better spacing
    plt.tight_layout()

    fig.savefig(f"{folder}/latent_rep_at_{rnd}.png")
    plt.close()
    return f"{folder}/latent_rep_at_{rnd}.png"


def visualize_plotly_latent_representation(
    model, test_loader, device, use_PCA=False, num_class=10
):
    model.eval()
    all_recons = []
    all_labels = []
    all_means = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            recon, mu, _ = model(data)
            all_recons.append(recon.cpu().numpy())
            all_means.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    all_recons = np.concatenate(all_recons, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    reduced_latents = all_means
    if use_PCA:

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        reduced_latents = pca.fit_transform(all_means)

    traces = []
    for label in np.unique(all_labels):
        indices = np.where(all_labels == label)
        trace = go.Scatter(
            x=reduced_latents[indices, 0].flatten(),
            y=reduced_latents[indices, 1].flatten(),
            mode="markers",
            name=f"Digit {label}",
            marker=dict(size=8),
        )
        traces.append(trace)

    layout = go.Layout(
        title="Latent Representation with True Labels",
        xaxis=dict(title="Principal Component 1"),
        yaxis=dict(title="Principal Component 2"),
        legend=dict(title="True Labels"),
        hovermode="closest",
    )
    fig = go.Figure(data=traces, layout=layout)

    return fig


def sample(net, device):
    """Generates samples usingfrom sklearn.mixture import GaussianMixture
    the decoder of the trained VAE."""
    with torch.no_grad():
        z = torch.randn(10)
        z = z.to(device)
        gen_image = net.decode(z)
    return gen_image


def sample_latents(model, test_loader, device, num_samples=64):
    model.eval()

    all_means = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            all_means.append(mu.cpu().numpy())

    all_means = np.concatenate(all_means, axis=0)
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=0)
    try:
        gmm.fit(all_means)
    except:
        print("Error in fitting GMM")
        return None
    return gmm.sample(num_samples)[0]


def get_fisher_ratio(model, test_loader, latent_dim, DEVICE):

    model.eval()

    latent_points = []
    labels_list = []
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        out, mu, _ = model(inputs)
        latent_points.append(mu)
        labels_list.append(labels)
    latent_points = torch.cat(latent_points, dim=0)
    labels = torch.cat(labels_list, dim=0)
    latent_points = latent_points.to(DEVICE)
    labels = labels.to(DEVICE)

    # Compute the mean of each class in the latent space
    latent_means = {}
    for class_label in torch.unique(labels):
        indices = torch.where(labels == class_label)[0]
        points = latent_points[indices]
        mean = torch.mean(points, dim=0)
        latent_means[class_label.item()] = mean

    # Separate the encoded points based on their corresponding classes
    latent_points_per_class = {}
    for class_label in torch.unique(labels):
        indices = torch.where(labels == class_label)[0]
        points = latent_points[indices]
        latent_points_per_class[class_label.item()] = points

    # Compute the within-class scatter matrix
    within_class_scatter = torch.zeros(latent_dim, latent_dim, device=DEVICE)
    for class_label, points in latent_points_per_class.items():
        mean = latent_means[class_label]
        centered_points = points - mean
        scatter_matrix = torch.matmul(centered_points.T, centered_points)
        within_class_scatter += scatter_matrix

    # Compute the between-class scatter matrix
    overall_mean = torch.mean(latent_points, dim=0)
    between_class_scatter = torch.zeros(latent_dim, latent_dim, device=DEVICE)
    for class_label, mean in latent_means.items():
        mean_diff = mean - overall_mean
        between_class_scatter += len(
            latent_points_per_class[class_label]
        ) * torch.outer(mean_diff, mean_diff)

    # Compute Fisher's linear discriminant ratio
    Fisher_ratio = torch.trace(
        torch.matmul(torch.inverse(within_class_scatter), between_class_scatter)
    )
    return Fisher_ratio.item()


def denormalize(tensor):
    # Adjust the normalization to be the inverse of what was applied to your dataset
    return tensor * 0.5 + 0.5


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def synthetic_noise_data(samples_per_class=100, num_labels=10):
    def generate_samples(label, num_samples):
        # Generate random noise for features
        features = np.random.rand(num_samples, 28, 28)
        # Normalize the features
        features = (features - 0.5) / 0.5
        # Create labels
        labels = np.full((num_samples,), label)
        return features, labels

    # Generate synthetic data for each label
    synthetic_data = []
    for label in range(num_labels):
        features, labels = generate_samples(label, samples_per_class)
        synthetic_data.append((features, labels))

    # Shuffle the synthetic data
    np.random.shuffle(synthetic_data)

    # Split data into features and labels
    all_features = np.concatenate([data[0] for data in synthetic_data], axis=0)
    all_labels = np.concatenate([data[1] for data in synthetic_data], axis=0)
    return (
        all_features.reshape(
            all_features.shape[0], 1, all_features.shape[-2], all_features.shape[-1]
        ),
        all_labels,
    )


class SyntheticDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def synthetic_alignment_dataloader(
    samples_per_class=100, num_labels=10, batch_size=1000, shuffle=True
):
    synthetic_features, synthetic_labels = synthetic_noise_data(
        samples_per_class, num_labels
    )
    synthetic_dataset = SyntheticDataset(synthetic_features, synthetic_labels)
    synthetic_dataloader = DataLoader(
        synthetic_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return synthetic_dataloader


if __name__ == "__main__":
    print(len(iid_train_iid_test()[0][1]))
    print(len(iid_train_iid_test()[0][2]))
    print(len(iid_train_iid_test()[0][3]))
    print(len(iid_train_iid_test()[0][4]))
    print(len(iid_train_iid_test()[0][0]))
