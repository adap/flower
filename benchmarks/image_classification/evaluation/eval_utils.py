import torch
from torchvision.transforms import ToTensor, Normalize, Compose


# transformation to convert images to tensors and apply normalization
def apply_transforms_test(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


# Test function
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * acc for num_examples, acc in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    weighted_acc = sum(accuracies) / sum(examples)

    return weighted_acc
