"""vitpoultry: Model, training, and dataset partitioning utilities."""

import torch
from datasets import load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def get_model(num_classes: int):
    """Return a pretrained ViT with all layers frozen except output head."""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, num_classes)

    model.requires_grad_(False)
    model.heads.requires_grad_(True)

    return model


def trainer(net, trainloader, optimizer, epochs, device: torch.device | str):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]
            loss.backward()
            optimizer.step()

    return total_loss / total_samples


def test(net, testloader, device: torch.device | str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


fds = None


def get_dataset_partition(num_partitions: int, partition_id: int, dataset_name: str):
    """Get dataset and partition it IID across clients (Simulation Engine)."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name, partitioners={"train": partitioner}
        )

    return fds.load_partition(partition_id)


def load_local_data(data_path: str, transform_fn):
    """Load a dataset partition from disk (Deployment Engine).

    Expects a HuggingFace dataset saved via `flwr-datasets create`.
    """
    dataset = load_from_disk(data_path)
    return dataset.with_transform(transform_fn)


def apply_eval_transforms(batch):
    """Apply standard evaluation transforms for ViT."""
    transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def apply_train_transforms(batch):
    """Apply standard training transforms with light augmentation."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


