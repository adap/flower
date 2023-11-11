import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from tqdm import tqdm
from torch.nn.parameter import Parameter
from typing import List, Tuple
import torch.nn as nn
# Define the root directory where your dataset is located
dataset_root = "data/pacs_data/"
from flwr.common import (
    bytes_to_ndarray,
)

# List the subdirectories (folders) in the parent folder
data_kinds = [
    os.path.join(dataset_root, d)
    for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
]


def make_dataloaders(dataset_kinds=data_kinds, k=2, batch_size=32, verbose=False):
    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    trainloaders = []
    valloaders = []
    test_subsets = []
    for idx, data_kind in enumerate(dataset_kinds):
        dataset = datasets.ImageFolder(root=data_kind, transform=transform)
        for fold, (train_indices, val_indices) in enumerate(
            skf.split(range(len(dataset)), dataset.targets), start=1
        ):
            # Split the dataset into train and val subsets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            if fold == 1:
                # Create DataLoaders for train and val sets
                train_loader_tmp = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader_tmp = DataLoader(val_dataset, batch_size=batch_size)
                if verbose:
                    print(
                        f"Fold {fold}: Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}"
                    )
                    val_labels = []

                    # Iterate through the val DataLoader to collect labels
                    for images, labels in val_loader_tmp:
                        val_labels.extend(labels.tolist())

                    # Calculate and print the class label frequency
                    label_counts = Counter(val_labels)

                    for label, count in label_counts.items():
                        print(f"Class {label}: {count} samples")

                trainloaders.append(train_loader_tmp)
                valloaders.append(val_loader_tmp)
            else:
                test_subsets.append(val_dataset)
    combined_test_dataset = ConcatDataset(test_subsets)
    testloader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=True)

    return (trainloaders, valloaders, testloader)


def train(net1, trainloader, optim, config, epochs, device: str, num_classes=7):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    lambda_reg = config.get("lambda_reg", 0.5)
    lambda_align = config.get("lambda_align", 5e-6)
    keys_to_check = ["z_g", "mu_g", "log_var_g"]

    # Pre-compute all_labels and check if z_g, mu_g, log_var_g are in config
    all_labels = torch.arange(num_classes).to(device)
    use_advanced_loss = all(key in config for key in keys_to_check)

    # Pre-compute tensors from config if using advanced loss
    if use_advanced_loss:
        z_g, mu_g, log_var_g = (
            torch.tensor(
                bytes_to_ndarray(config["z_g"]), dtype=torch.float32, device=device
            ),
            torch.tensor(
                bytes_to_ndarray(config["mu_g"]), dtype=torch.float32, device=device
            ),
            torch.tensor(
                bytes_to_ndarray(config["log_var_g"]),
                dtype=torch.float32,
                device=device,
            ),
        )

    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optim.zero_grad()
            images, labels = images.to(device), labels.to(device)
            pred, mu, log_var = net1(images)

            # loss_fl
            loss_fl = criterion(pred, labels)
            loss = loss_fl

            if use_advanced_loss:
                # loss_reg
                loss_reg = criterion(net1.clf(z_g), all_labels)

                # KL Div
                loss_align = 0.5 * (log_var_g[labels] - log_var - 1) + (
                    log_var.exp() + (mu - mu_g[labels]).pow(2)
                ) / (2 * log_var_g[labels].exp())
                loss_align_reduced = loss_align.mean(dim=1).mean()
                loss += lambda_reg * loss_reg + lambda_align * loss_align_reduced

            loss.backward(retain_graph=use_advanced_loss)
            optim.step()

def train_prox(  
    net,
    trainloader,
    optim,
    config,
    epochs,
    device,
    num_classes,
):

    criterion = torch.nn.CrossEntropyLoss()
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, global_params, trainloader, device, criterion, optim, config.get("proximal_mu", 1)
        )


def _train_one_epoch(  
    net,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        print(type(net(images)))
        print(type(labels))
        loss.backward()
        optimizer.step()
    return net

def test(net1, testloader, device: str):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net1(images.to(device))[0]
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


if __name__ == "__main__":
    aq, bq, cq = make_dataloaders(verbose=True)

    cc = 0
    for feat, label in cq:
        print(len(label))
        cc += len(label)
