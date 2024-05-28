import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torchvision.models import efficientnet_b0, AlexNet
import warnings

from flwr_datasets import FederatedDataset


warnings.filterwarnings("ignore")


def load_partition(partition_id, toy: bool = False):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    return partition_train_test["train"], partition_train_test["test"]


def load_centralized_data():
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    centralized_data = fds.load_split("test")
    centralized_data = centralized_data.with_transform(apply_transforms)
    return centralized_data


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def train(
    net, trainloader, valloader, epochs, device: torch.device = torch.device("cpu")
):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(
    net, testloader, steps: int = None, device: torch.device = torch.device("cpu")
):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


def load_efficientnet(classes: int = 10):
    """Loads EfficienNetB0 from TorchVision."""
    efficientnet = efficientnet_b0(pretrained=True)
    # Re-init output linear layer with the right number of classes
    model_classes = efficientnet.classifier[1].in_features
    if classes != model_classes:
        efficientnet.classifier[1] = torch.nn.Linear(model_classes, classes)
    return efficientnet


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def load_alexnet(classes):
    """Load AlexNet model from TorchVision."""
    return AlexNet(num_classes=classes)
